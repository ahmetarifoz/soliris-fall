"""Entrypoint for the fall detection runtime."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2

from . import logger
from .alert import send_op500_trigger, send_webhook
from .api import set_detector, start_api_thread
from .config import AppSection, Config, load_config, resolve_stream_ingest
from .detector import FallDetector
from .hud import HUD
from .stream import StreamReader
from .webhook import WebhookClient

WINDOW_NAME = "Fall Detector"
ENV_CAPTURE_OPTIONS = "OPENCV_FFMPEG_CAPTURE_OPTIONS"


def _resolve_source(source: Any) -> Any:
    if isinstance(source, str):
        if source.startswith("webcam:") or source.isdigit():
            index = int(source.split(":")[1]) if ":" in source else int(source)
            return index
        return source
    return source


def _is_rtsp_source(source: Any) -> bool:
    return isinstance(source, str) and source.lower().startswith("rtsp://")


def _is_sdp_source(source: Any) -> bool:
    """Check if source is an SDP file (RTP stream)."""
    if not isinstance(source, str):
        return False
    return source.lower().endswith(".sdp") or source.lower().startswith("sdp://")

def _build_ffmpeg_options(app_cfg: AppSection) -> Optional[str]:
    # Mevcut config'ten kopyala (None gelirse boş dict olsun)
    opts = dict(app_cfg.rtsp_ffmpeg_options or {})

    # rtsp_transport yoksa config'ten ekle
    if app_cfg.rtsp_transport and "rtsp_transport" not in opts:
        opts["rtsp_transport"] = app_cfg.rtsp_transport

    # RTP / SDP için zorunlu whitelist
    # (FFmpeg varsayılan: file,crypto,data -> buraya rtp,udp,tcp ekliyoruz)
    default_whitelist = "file,rtp,udp,tcp,crypto,data"
    if "protocol_whitelist" in opts:
        # Config'te varsa, required ile merge et
        current = set(opts["protocol_whitelist"].split(","))
        required = set(default_whitelist.split(","))
        merged = ",".join(sorted(current | required))
        opts["protocol_whitelist"] = merged
    else:
        opts["protocol_whitelist"] = default_whitelist

    if not opts:
        return None

    # OPENCV_FFMPEG_CAPTURE_OPTIONS formatı: key=value|key2=value2|...
    return "|".join(f"{key}={value}" for key, value in opts.items())



def _create_capture(
    source: Any,
    backend: str,
    *,
    is_rtsp: bool,
    is_sdp: bool = False,
    ffmpeg_options: Optional[str],
) -> cv2.VideoCapture:
    previous_env = None
    applied_options = False
    use_ffmpeg = backend.lower() == "ffmpeg"
    needs_ffmpeg_opts = (is_rtsp or is_sdp) and use_ffmpeg and ffmpeg_options
    
    if needs_ffmpeg_opts:
        previous_env = os.environ.get(ENV_CAPTURE_OPTIONS)
        os.environ[ENV_CAPTURE_OPTIONS] = ffmpeg_options
        applied_options = True
    try:
        cap = (
            cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if needs_ffmpeg_opts
            else cv2.VideoCapture(source)
        )
    finally:
        if applied_options:
            if previous_env is None:
                os.environ.pop(ENV_CAPTURE_OPTIONS, None)
            else:
                os.environ[ENV_CAPTURE_OPTIONS] = previous_env
    if (is_rtsp or is_sdp) and cap.isOpened():
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _ensure_writer(cfg: Config, capture: cv2.VideoCapture) -> Tuple[Any, float]:
    fps = capture.get(cv2.CAP_PROP_FPS) or cfg.app.default_fps
    writer = None
    if cfg.app.save_path:
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        path = Path(cfg.app.save_path)
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    return writer, fps


def _read_frame(
    cap: cv2.VideoCapture,
    *,
    drop_frames: int,
    is_rtsp: bool,
) -> Tuple[bool, Optional[Any]]:
    if is_rtsp and drop_frames > 0:
        for _ in range(drop_frames):
            if not cap.grab():
                return False, None
    return cap.read()


def _create_disabled_frame(width: int, height: int):
    """Create a blank frame showing detection disabled state."""
    import numpy as np
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)  # Dark gray background
    
    # Draw "DETECTION DISABLED" text
    text = "DETECTION DISABLED"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (100, 100, 100), thickness, cv2.LINE_AA)
    
    # Draw instruction
    instruction = "Enable via API: POST /enable"
    inst_scale = 0.8
    inst_size = cv2.getTextSize(instruction, font, inst_scale, 2)[0]
    inst_x = (width - inst_size[0]) // 2
    inst_y = text_y + 50
    cv2.putText(frame, instruction, (inst_x, inst_y), font, inst_scale, (80, 80, 80), 2, cv2.LINE_AA)
    
    return frame


def run(config_path: str | Path = "config.yaml") -> None:
    cfg = load_config(config_path)

    # Get first stream config (like fsdapp)
    stream_cfg = cfg.streams[0]
    source = _resolve_source(stream_cfg.source)
    is_rtsp = _is_rtsp_source(source)
    is_sdp = _is_sdp_source(source)
    ffmpeg_options = _build_ffmpeg_options(cfg.app)
    
    # Resolve effective ingest settings (like fsdapp)
    ingest_settings = resolve_stream_ingest(cfg, stream_cfg)
    
    # Lazy initialization - StreamReader/VideoCapture created when detection enabled
    # SDP/RTP sources require StreamReader (FFmpeg subprocess) for proper protocol_whitelist support
    use_stream_reader = stream_cfg.rtp_loopback_enabled or is_sdp
    sr: Optional[StreamReader] = None
    cap: Optional[cv2.VideoCapture] = None
    fps = cfg.app.default_fps
    writer = None
    
    # Store config for lazy init
    _stream_init_done = False
    
    def _init_stream():
        """Initialize stream reader or video capture (called when detection enabled)."""
        nonlocal sr, cap, fps, writer, _stream_init_done
        if _stream_init_done:
            return True
        
        if use_stream_reader:
            sr_new = StreamReader(
                source=stream_cfg.source,
                reconnect_delay=ingest_settings["reconnect_seconds"],
                width=ingest_settings["width"],
                height=ingest_settings["height"],
                backend=ingest_settings["backend"],
                ffmpeg_hw=ingest_settings["ffmpeg_hw"],
                queue_size=ingest_settings["queue_size"],
                rtp_output_port=stream_cfg.rtp_port,
            )
            # Configure OP500 if enabled
            if cfg.op500 and cfg.op500.enabled:
                sr_new.op500_enabled = True
                sr_new.op500_base_url = cfg.op500.base_url
            sr = sr_new
            # Link to detector
            detector.stream_reader = sr
            logger.info(f"StreamReader initialized for {stream_cfg.source}")
        else:
            cap_new = _create_capture(
                source,
                cfg.app.rtsp_backend,
                is_rtsp=is_rtsp,
                is_sdp=is_sdp,
                ffmpeg_options=ffmpeg_options,
            )
            if not cap_new.isOpened():
                logger.error(f"Cannot open source: {stream_cfg.source}")
                return False
            cap = cap_new
            fps = cap.get(cv2.CAP_PROP_FPS) or cfg.app.default_fps
            writer_new, fps = _ensure_writer(cfg, cap)
            if writer_new:
                writer = writer_new
            logger.info(f"VideoCapture initialized for {stream_cfg.source}")
        
        _stream_init_done = True
        return True

    # Use camera_id from stream config, fallback to source
    camera_id = stream_cfg.camera_id or stream_cfg.source
    detector = FallDetector(cfg, camera_id=camera_id)
    
    # Configure OP500 on detector
    if cfg.op500 and cfg.op500.enabled:
        detector.op500_enabled = True
        detector.op500_base_url = cfg.op500.base_url
        detector.op500_rtp_auto_close_seconds = cfg.op500.rtp_auto_close_seconds
        if stream_cfg.rtp_port:
            detector.op500_rtp_port = stream_cfg.rtp_port
    
    hud = HUD(max_tracks=cfg.app.max_tracks) if cfg.app.enable_hud else None
    webhook = WebhookClient(cfg.webhook)

    # Configure HUD with RTP and cooldown settings
    if hud:
        hud.set_alert_cooldown(cfg.detect.cooldown_sec)
        if stream_cfg.rtp_port:
            hud.rtp_port = stream_cfg.rtp_port

    frame_skip = max(0, cfg.detect.frame_skip)
    skip_cursor = 0

    if hud:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    # Start API server if enabled
    if cfg.api and cfg.api.enabled:
        set_detector(detector, cfg)
        start_api_thread(host=cfg.api.host, port=cfg.api.port)
        logger.info(f"API server started on http://{cfg.api.host}:{cfg.api.port}")

    logger.info(f"Starting fall detection on {stream_cfg.source}")
    logger.info("Detection starts DISABLED. Enable via API POST /enable")
    logger.info("Press ESC to quit.")

    async def maybe_alert(event, trigger_op500: bool = False):
        """Send webhook notification for detection event."""
        payload = event.to_payload()
        await send_webhook(cfg.webhook.url, payload, cfg.webhook.hmac_secret or "")

    try:
        # Use StreamReader frames generator or OpenCV capture
        frame_source = None  # Lazy init - only start when detection enabled
        
        while True:
            # Wait while detection is disabled (don't read from stream)
            if not detector.detection_enabled:
                # Show disabled state in HUD
                if hud:
                    cv2.imshow(WINDOW_NAME, _create_disabled_frame(1280, 720))
                    if cv2.waitKey(100) & 0xFF == 27:
                        break
                else:
                    time.sleep(0.1)
                continue
            
            # Lazy init stream when detection becomes enabled
            if not _stream_init_done:
                if not _init_stream():
                    logger.error("Failed to initialize stream, retrying...")
                    time.sleep(1.0)
                    continue
            
            # Lazy init frame source when detection becomes enabled
            if frame_source is None and sr:
                frame_source = sr.frames()
            
            if sr:
                try:
                    frame = next(frame_source)
                except StopIteration:
                    break
            else:
                ok, frame = _read_frame(
                    cap,
                    drop_frames=cfg.app.rtsp_drop_frames,
                    is_rtsp=is_rtsp,
                )
                if not ok or frame is None:
                    if (is_rtsp or is_sdp) and cfg.app.rtsp_reconnect:
                        logger.stream_disconnected(stream_cfg.source)
                        cap.release()
                        reconnected = False
                        for attempt in range(1, cfg.app.rtsp_max_retries + 1):
                            logger.stream_reconnecting(stream_cfg.source, attempt)
                            time.sleep(cfg.app.rtsp_retry_delay)
                            cap = _create_capture(
                                source,
                                cfg.app.rtsp_backend,
                                is_rtsp=is_rtsp,
                                is_sdp=is_sdp,
                                ffmpeg_options=ffmpeg_options,
                            )
                            if cap.isOpened():
                                fps = cap.get(cv2.CAP_PROP_FPS) or cfg.app.default_fps
                                if writer is None and cfg.app.save_path:
                                    writer, fps = _ensure_writer(cfg, cap)
                                reconnected = True
                                skip_cursor = 0
                                logger.stream_connected(stream_cfg.source)
                                break
                        if not reconnected:
                            logger.error(f"RTSP reconnect attempts exhausted for {stream_cfg.source}")
                            break
                        continue
                    logger.error("Frame could not be read; exiting loop.")
                    break
            
            if frame is None:
                continue

            if frame_skip > 0:
                skip_cursor = (skip_cursor + 1) % (frame_skip + 1)
                if skip_cursor:
                    continue

            result = detector.process_frame(frame, fps)

            for event in result.events:
                # Legacy webhook
                webhook.send(event)
                
                # RTP + OP500 flow (strict order per business rules):
                # 1. If RTP not active: sender/add → FFmpeg start → OP500 trigger
                # 2. If RTP active: extend duration, NO trigger
                if stream_cfg.rtp_loopback_enabled and sr and stream_cfg.rtp_port:
                    # trigger_rtp_stream returns True if newly started
                    is_new_rtp_start = sr.trigger_rtp_stream()
                    
                    # OP500 trigger ONLY on new RTP start (not on extend)
                    if is_new_rtp_start and cfg.op500 and cfg.op500.enabled:
                        asyncio.run(send_op500_trigger(
                            base_url=cfg.op500.base_url,
                            port=stream_cfg.rtp_port,
                            event_type="FALL",
                            delay_sec=5
                        ))
                
                # Async webhook (always send)
                asyncio.run(maybe_alert(event, trigger_op500=False))

            # Update HUD with RTP state and cooldowns
            if hud:
                # Update RTP status
                if sr:
                    hud.set_rtp_state(
                        active=getattr(sr, '_rtp_active', False),
                        port=stream_cfg.rtp_port
                    )
                
                # Update per-track cooldowns from detector
                now = time.time()
                for tid, last_alert_time in detector.last_alert.items():
                    remaining = cfg.detect.cooldown_sec - (now - last_alert_time)
                    hud.set_cooldown(tid, remaining)

            # Send frame with detections to RTP if streaming is active
            if sr and stream_cfg.rtp_loopback_enabled:
                vis = result.frame
                if hud:
                    hud.draw(vis, result.tracks_view)
                sr.send_frame_to_rtp(vis)

            if hud:
                hud.draw(result.frame, result.tracks_view)
                cv2.imshow(WINDOW_NAME, result.frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if writer is not None:
                writer.write(result.frame)
    finally:
        if sr:
            sr.close()
        if cap:
            cap.release()
        if writer is not None:
            writer.release()
        if hud:
            cv2.destroyWindow(WINDOW_NAME)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
