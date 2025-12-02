"""Core fall detection pipeline built on top of YOLO pose tracking."""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from . import logger
from .config import Config

COCO_EDGES = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 12),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


@dataclass
class FallEvent:
    id: str
    track_id: int
    angle_mean: float
    velocity_blps: float
    drop_rel: float
    aspect_ratio: float
    timestamp: float
    status: str
    camera_id: str
    confidence: float

    def to_payload(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "track_id": int(self.track_id),
            "angle_mean": float(self.angle_mean),
            "velocity_blps": float(self.velocity_blps),
            "drop_rel": float(self.drop_rel),
            "aspect_ratio": float(self.aspect_ratio),
            "timestamp": float(self.timestamp),
            "status": self.status,
            "camera_id": self.camera_id,
            "confidence": float(self.confidence),
        }


@dataclass
class FrameResult:
    frame: np.ndarray
    tracks_view: Dict[int, Dict[str, Any]]
    events: List[FallEvent]


class FallDetector:
    def __init__(self, config: Config, camera_id: Optional[str] = None):
        self.cfg = config
        self.camera_id = camera_id or config.streams[0].camera_id or config.streams[0].source
        self.model = YOLO(config.app.model_path)
        self.buffers: Dict[int, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.cfg.detect.temporal_window)
        )
        self.track_state: Dict[int, Dict[str, Any]] = {}
        self.last_seen: Dict[int, float] = {}
        self.alert_until: Dict[int, float] = {}
        
        # Per-track cooldown timers (like fsdapp's per-class cooldown)
        self.last_alert: Dict[int, float] = {}
        
        # OP500 integration
        self.op500_enabled: bool = False
        self.op500_base_url: Optional[str] = None
        self.op500_rtp_port: Optional[int] = None
        self.op500_rtp_auto_close_seconds: int = 120
        
        # RTP loopback state tracking
        self.rtp_loopback_active: bool = False
        self.last_detection_time: float = 0.0
        
        # Detection control flag (starts disabled, enable via API)
        self.detection_enabled: bool = False
        
        # StreamReader reference (will be set externally)
        self.stream_reader = None

    @staticmethod
    def _torso_angle(kp17: np.ndarray) -> float:
        ls, rs, lh, rh = 5, 6, 11, 12
        shoulder = (kp17[ls] + kp17[rs]) / 2.0
        hip = (kp17[lh] + kp17[rh]) / 2.0
        vec = shoulder - hip
        return float(np.degrees(np.arctan2(abs(vec[1]), abs(vec[0]) + 1e-6)))

    @staticmethod
    def _floor_fraction(hip_y: float, bbox: np.ndarray) -> float:
        y_bottom = bbox[3]
        height = max(1.0, bbox[3] - bbox[1])
        return max(0.0, min(1.0, (y_bottom - hip_y) / height))

    @staticmethod
    def _body_scale(kp17: np.ndarray) -> float:
        return float(np.linalg.norm(kp17[5] - kp17[11]) + 1e-6)

    def _standing_like(self, kp17: np.ndarray) -> bool:
        cfg = self.cfg.detect
        lhip, rhip = kp17[11], kp17[12]
        lknee, rknee = kp17[13], kp17[14]
        hip_y = (lhip[1] + rhip[1]) / 2
        knee_y = (lknee[1] + rknee[1]) / 2
        return (self._torso_angle(kp17) > cfg.recover_angle) and (knee_y > hip_y)

    def _draw_pose(self, frame: np.ndarray, kp17: np.ndarray) -> None:
        for x, y in kp17.astype(int):
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), -1)
        for a, b in COCO_EDGES:
            xa, ya = kp17[a]
            xb, yb = kp17[b]
            cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), (255, 200, 0), 2)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        boxes_xyxy: Optional[np.ndarray],
        ids: List[int],
        angles: List[float],
        kps_list: List[np.ndarray],
        debug_map: Dict[int, Dict[str, float]],
    ) -> None:
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return
        now = time.time()
        for bbox, tid, angle, kp in zip(boxes_xyxy, ids, angles, kps_list):
            x1, y1, x2, y2 = map(int, bbox.tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 220, 60), 2)
            label = f"id:{int(tid)} ang:{angle:.1f}"
            if tid in debug_map:
                dbg = debug_map[tid]
                label += (
                    f" v:{dbg['v']:.2f} dr:{dbg['dr']:.2f}"
                    f" ar:{dbg['ar']:.2f} lift:{dbg['lift']:.2f}"
                )
            cv2.putText(
                frame,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (60, 220, 60),
                2,
                cv2.LINE_AA,
            )
            self._draw_pose(frame, kp)
            if tid in self.alert_until and now < self.alert_until[tid]:
                cv2.putText(
                    frame,
                    "Fall Detected",
                    (x1, y1 - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

    def _start_rtp_loopback(self) -> bool:
        """Start RTP loopback - returns True if newly started, False if already active."""
        if self.stream_reader and getattr(self.stream_reader, '_rtp_active', False):
            self.rtp_loopback_active = True
            return False  # Already active

        if not self.op500_enabled or not self.op500_base_url or not self.op500_rtp_port:
            return False

        if not self.stream_reader:
            return False

        try:
            self.stream_reader._start_rtp_sender()
            if getattr(self.stream_reader, '_rtp_active', False) and self.stream_reader._rtp_proc:
                print(f"RTP loopback started (port={self.op500_rtp_port})")
                self.rtp_loopback_active = True
                return True
            else:
                self.rtp_loopback_active = False
                return False
        except Exception as e:
            print(f"RTP loopback start error: {e}")
            self.rtp_loopback_active = False
            return False

    def _stop_rtp_loopback(self):
        """Stop RTP loopback via OP500 sender API DELETE."""
        if not self.op500_enabled or not self.op500_base_url or not self.op500_rtp_port:
            return

        if not self.rtp_loopback_active:
            return

        try:
            import requests
            url = f"{self.op500_base_url}detectors/sender/{self.op500_rtp_port}"
            print(f"Stopping RTP loopback: DELETE {url}")
            response = requests.delete(url, timeout=4)

            if response.status_code in (200, 204):
                self.rtp_loopback_active = False
                if self.stream_reader:
                    self.stream_reader._stop_rtp_sender()
                print(f"RTP loopback stopped (port={self.op500_rtp_port})")
            else:
                self.rtp_loopback_active = False
        except Exception as e:
            print(f"RTP loopback stop error: {e}")

    def _check_rtp_auto_close(self):
        """Check if RTP loopback should auto-close due to no recent detections."""
        if not self.rtp_loopback_active:
            return

        elapsed = time.time() - self.last_detection_time

        if elapsed > self.op500_rtp_auto_close_seconds:
            print(f"RTP auto-close triggered: no detection for {elapsed:.0f}s")
            self._stop_rtp_loopback()

    def process_frame(self, frame: np.ndarray, fps: float) -> FrameResult:
        # Check if detection is enabled
        if not self.detection_enabled:
            return FrameResult(frame=frame, tracks_view={}, events=[])

        # RTP loopback auto-close check
        self._check_rtp_auto_close()

        cfg = self.cfg.detect
        results = self.model.track(
            frame,
            device=self.cfg.app.device,
            conf=cfg.conf,
            iou=cfg.iou,
            verbose=False,
            persist=cfg.track_persist,
            tracker="botsort.yaml",
        )

        now = time.time()
        ids: List[int] = []
        boxes: List[np.ndarray] = []
        angles: List[float] = []
        kps_list: List[np.ndarray] = []
        confidences: Dict[int, float] = {}

        for res in results:
            if res.keypoints is None or res.boxes is None or res.boxes.id is None:
                continue
            karr = res.keypoints.xy.cpu().numpy()
            barr = res.boxes.xyxy.cpu().numpy()
            iarr = res.boxes.id.cpu().numpy().astype(int)
            confarr = res.boxes.conf.cpu().numpy() if res.boxes.conf is not None else None
            for j, tid in enumerate(iarr):
                kp17 = karr[j][:17]
                angle = self._torso_angle(kp17)
                hip = (kp17[11] + kp17[12]) / 2.0
                bbox = barr[j]
                self.buffers[tid].append(
                    {"kp": kp17, "hip": hip, "ang": angle, "ts": now, "bb": bbox}
                )
                ids.append(tid)
                self.last_seen[tid] = now
                boxes.append(bbox)
                angles.append(angle)
                kps_list.append(kp17)
                if confarr is not None:
                    confidences[tid] = float(confarr[j])

        events: List[FallEvent] = []
        debug_map: Dict[int, Dict[str, float]] = {}

        for tid, dq in list(self.buffers.items()):
            if len(dq) < cfg.temporal_min_hits:
                continue

            st = self.track_state.get(
                tid,
                {
                    "state": "idle",
                    "t_enter": now,
                    "t_last": now,
                    "candidate_notified": False,
                    "fallen_notified": False,
                },
            )
            st.setdefault("candidate_notified", False)
            st.setdefault("fallen_notified", False)
            state = st["state"]
            recent = list(dq)[-cfg.temporal_min_hits :]
            ang_mean = float(np.mean([x["ang"] for x in recent]))

            scale_now = self._body_scale(recent[-1]["kp"])
            hips = np.stack([x["hip"] for x in dq], axis=0)
            pix_per_frame = np.linalg.norm(
                np.diff(hips, axis=0, prepend=hips[:1]), axis=1
            )
            v_blps_series = (pix_per_frame * fps) / max(scale_now, 1e-6)
            v_blps_mean = float(np.mean(v_blps_series[-cfg.temporal_min_hits :]))

            k = int(max(1, round(0.5 * fps)))
            y_now = float(recent[-1]["hip"][1])
            y_prev = float(dq[-k]["hip"][1]) if len(dq) > k else float(dq[0]["hip"][1])
            drop_rel = float((y_now - y_prev) / max(scale_now, 1e-6))

            bb = dq[-1]["bb"]
            bw = max(1.0, bb[2] - bb[0])
            bh = max(1.0, bb[3] - bb[1])
            aspect_ratio = float(bw / bh)
            floor_frac = self._floor_fraction(y_now, bb)

            pelvis_now = y_now
            if "pelvis_min" not in st:
                st["pelvis_min"] = pelvis_now
            if state == "fallen":
                st["pelvis_min"] = max(st.get("pelvis_min", pelvis_now), pelvis_now)

            still = v_blps_mean < cfg.still_bls
            fall_cond = (
                (ang_mean < cfg.angle_th and still)
                or (drop_rel > cfg.drop_rel and ang_mean < cfg.angle_th + 8)
                or (ang_mean < cfg.angle_th + 5 and aspect_ratio > 1.25)
                or (floor_frac < cfg.pelvis_floor_frac and ang_mean < cfg.angle_th + 6)
            )
            lift_rel = (st.get("pelvis_min", pelvis_now) - pelvis_now) / max(
                scale_now, 1e-6
            )
            true_recover = (ang_mean > cfg.recover_angle) and (
                lift_rel >= cfg.recover_lift_rel or self._standing_like(recent[-1]["kp"])
            )

            conf = confidences.get(tid, cfg.conf)

            if state == "idle":
                if fall_cond:
                    state = "candidate"
                    st["t_enter"] = now
                    st["candidate_notified"] = False
                    st["fallen_notified"] = False
            elif state == "candidate":
                if fall_cond and (now - st["t_enter"] >= cfg.confirm_sec):
                    state = "fallen"
                    st["t_enter"] = now
                    st["t_alarm"] = now
                    st["pelvis_min"] = pelvis_now
                    st["fallen_notified"] = False
                elif not fall_cond:
                    state = "idle"
                    st["t_enter"] = now
                    st["candidate_notified"] = False
                    st["fallen_notified"] = False
                elif (
                    now - st.get("t_enter", now) >= cfg.candidate_alert_sec
                    and not st.get("candidate_notified", False)
                ):
                    # Check per-track cooldown (like fsdapp's per-class cooldown)
                    last_alert_time = self.last_alert.get(tid, 0.0)
                    if now - last_alert_time < cfg.cooldown_sec:
                        remaining = cfg.cooldown_sec - (now - last_alert_time)
                        logger.alert_cooldown(tid, remaining)
                        continue  # Still in cooldown, skip
                    
                    # Start RTP loopback on detection
                    rtp_just_started = self._start_rtp_loopback()
                    self.last_detection_time = now
                    
                    # Log candidate alert
                    logger.alert_candidate(tid, self.camera_id, ang_mean, conf)
                    
                    event = FallEvent(
                        id=str(uuid.uuid4()),
                        track_id=int(tid),
                        angle_mean=float(round(float(ang_mean), 2)),
                        velocity_blps=float(round(float(v_blps_mean), 3)),
                        drop_rel=float(round(float(drop_rel), 3)),
                        aspect_ratio=float(round(float(aspect_ratio), 3)),
                        timestamp=float(now),
                        status="candidate",
                        camera_id=str(self.camera_id),
                        confidence=float(conf),
                    )
                    # Add RTP trigger flag to event payload
                    event._trigger_op500 = rtp_just_started
                    events.append(event)
                    self.last_alert[tid] = now
                    st["candidate_notified"] = True
            elif state == "fallen":
                elapsed_fallen = now - st.get("t_alarm", st.get("t_enter", now))
                latch_ok = elapsed_fallen >= max(cfg.cooldown_sec, cfg.fall_latch_sec)
                if (
                    not st.get("fallen_notified", False)
                    and elapsed_fallen >= cfg.fallen_alert_sec
                ):
                    # Check per-track cooldown
                    last_alert_time = self.last_alert.get(tid, 0.0)
                    if now - last_alert_time >= cfg.cooldown_sec:
                        # Start RTP loopback on detection
                        rtp_just_started = self._start_rtp_loopback()
                        self.last_detection_time = now
                        
                        # Log fall detection
                        logger.alert_fallen(tid, self.camera_id, ang_mean, conf)
                        
                        event = FallEvent(
                            id=str(uuid.uuid4()),
                            track_id=int(tid),
                            angle_mean=float(round(float(ang_mean), 2)),
                            velocity_blps=float(round(float(v_blps_mean), 3)),
                            drop_rel=float(round(float(drop_rel), 3)),
                            aspect_ratio=float(round(float(aspect_ratio), 3)),
                            timestamp=float(now),
                            status="fallen",
                            camera_id=str(self.camera_id),
                            confidence=float(conf),
                        )
                        event._trigger_op500 = rtp_just_started
                        events.append(event)
                        self.alert_until[tid] = now + cfg.alert_hold
                        self.last_alert[tid] = now
                        st["fallen_notified"] = True
                if latch_ok and true_recover:
                    state = "recover"
                    st["t_enter"] = now
                    st["candidate_notified"] = False
                    st["fallen_notified"] = False
            elif state == "recover":
                if true_recover and (now - st["t_enter"] >= cfg.recover_sec):
                    # Log recovery
                    logger.alert_recovered(tid, self.camera_id)
                    state = "idle"
                    st["t_enter"] = now
                    st["candidate_notified"] = False
                    st["fallen_notified"] = False
                elif not true_recover:
                    state = "fallen"
                    st.setdefault("t_alarm", now)

            st["state"] = state
            st["t_last"] = now
            self.track_state[tid] = st
            debug_map[tid] = {
                "v": v_blps_mean,
                "dr": drop_rel,
                "ar": aspect_ratio,
                "lift": lift_rel,
            }

        if ids:
            self._draw_overlay(
                frame, np.array(boxes), ids, angles, kps_list, debug_map
            )

        tracks_view: Dict[int, Dict[str, Any]] = {}
        current_ids = set(ids)
        for tid in list(self.buffers.keys()):
            last_seen = self.last_seen.get(tid, now)
            if tid not in current_ids and (now - last_seen) > cfg.prune_sec:
                self.buffers.pop(tid, None)
                self.track_state.pop(tid, None)
                self.alert_until.pop(tid, None)
                self.last_seen.pop(tid, None)
                continue
            state_info = self.track_state.get(tid, {"state": "idle", "t_enter": now})
            buffer_items = list(self.buffers.get(tid, []))
            angle_mean = (
                float(np.mean([x["ang"] for x in buffer_items[-cfg.temporal_min_hits :]]))
                if len(buffer_items) >= cfg.temporal_min_hits
                else 0.0
            )
            dbg = debug_map.get(
                tid, {"v": 0.0, "dr": 0.0, "ar": 0.0, "lift": 0.0}
            )
            bbox = (
                buffer_items[-1]["bb"]
                if buffer_items and "bb" in buffer_items[-1]
                else None
            )
            tracks_view[tid] = {
                "state": state_info.get("state", "idle"),
                "since": state_info.get("t_enter", now),
                "angle": angle_mean,
                "v": dbg["v"],
                "drop": dbg["dr"],
                "ar": dbg["ar"],
                "box": bbox,
                "alert": (tid in self.alert_until and now < self.alert_until[tid]),
                "last_seen": last_seen,
                "confidence": confidences.get(tid, cfg.conf),
            }

        return FrameResult(frame=frame, tracks_view=tracks_view, events=events)


__all__ = ["FallDetector", "FallEvent", "FrameResult"]
