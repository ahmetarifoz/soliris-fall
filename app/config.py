"""Structured configuration loading for the fall detection app."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass(frozen=True)
class AppSection:
    model_path: str
    device: Any
    enable_hud: bool
    save_path: Optional[str]
    default_fps: float
    max_tracks: int
    rtsp_reconnect: bool
    rtsp_max_retries: int
    rtsp_retry_delay: float
    rtsp_backend: str
    rtsp_transport: str
    rtsp_ffmpeg_options: Dict[str, str]
    rtsp_drop_frames: int


@dataclass(frozen=True)
class DetectSection:
    conf: float
    iou: float
    track_persist: bool
    confirm_sec: float
    recover_sec: float
    recover_angle: float
    cooldown_sec: float
    alert_hold: float
    temporal_window: int
    temporal_min_hits: int
    angle_th: float
    still_bls: float
    drop_rel: float
    prune_sec: float
    fall_latch_sec: float
    pelvis_floor_frac: float
    recover_lift_rel: float
    frame_skip: int
    candidate_alert_sec: float
    fallen_alert_sec: float


@dataclass(frozen=True)
class WebhookSection:
    url: str
    hmac_secret: Optional[str]
    timeout_seconds: float
    verify_ssl: bool
    cooldown_seconds: float


@dataclass(frozen=True)
class Config:
    app: AppSection
    detect: DetectSection
    webhook: WebhookSection
    streams: List[str]


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must define a mapping at the top level")
    return data


def _extract_ffmpeg_options(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    options: Dict[str, str] = {}
    for key, value in raw.items():
        options[str(key)] = str(value)
    return options


def load_config(path: str | Path = "config.yaml") -> Config:
    base_path = Path(path)
    data = _load_yaml(base_path)

    local_path = base_path.with_name(base_path.stem + ".local" + base_path.suffix)
    if local_path.exists():
        local_data = _load_yaml(local_path)
        data = _deep_merge(data, local_data)

    app_section = data.get("app", {})
    detect_section = data.get("detect", {})
    webhook_section = data.get("webhook", {})
    streams_section = data.get("streams", [])

    if not streams_section:
        raise ValueError("streams list cannot be empty; add at least one source in config.yaml")

    app = AppSection(
        model_path=str(app_section.get("model_path", "models/yolo11m-pose.pt")),
        device=app_section.get("device", "cpu"),
        enable_hud=bool(app_section.get("enable_hud", True)),
        save_path=app_section.get("save_path") or None,
        default_fps=float(app_section.get("default_fps", 60.0)),
        max_tracks=int(app_section.get("max_tracks", 6)),
        rtsp_reconnect=bool(app_section.get("rtsp_reconnect", True)),
        rtsp_max_retries=int(app_section.get("rtsp_max_retries", 5)),
        rtsp_retry_delay=float(app_section.get("rtsp_retry_delay", 2.5)),
        rtsp_backend=str(app_section.get("rtsp_backend", "ffmpeg")),
        rtsp_transport=str(app_section.get("rtsp_transport", "tcp")),
        rtsp_ffmpeg_options=_extract_ffmpeg_options(app_section.get("rtsp_ffmpeg_options", {})),
        rtsp_drop_frames=int(app_section.get("rtsp_drop_frames", 3)),
    )

    detect = DetectSection(
        conf=float(detect_section.get("conf", 0.25)),
        iou=float(detect_section.get("iou", 0.55)),
        track_persist=bool(detect_section.get("track_persist", True)),
        confirm_sec=float(detect_section.get("confirm_sec", 0.2)),
        recover_sec=float(detect_section.get("recover_sec", 0.7)),
        recover_angle=float(detect_section.get("recover_angle", 45.0)),
        cooldown_sec=float(detect_section.get("cooldown_sec", 0.4)),
        alert_hold=float(detect_section.get("alert_hold", 4.0)),
        temporal_window=int(detect_section.get("temporal_window", detect_section.get("win", 60))),
        temporal_min_hits=int(detect_section.get("temporal_min_hits", detect_section.get("required", 3))),
        angle_th=float(detect_section.get("angle_th", 46.0)),
        still_bls=float(detect_section.get("still_bls", 0.12)),
        drop_rel=float(detect_section.get("drop_rel", 1.0)),
        prune_sec=float(detect_section.get("prune_sec", 1.2)),
        fall_latch_sec=float(detect_section.get("fall_latch_sec", 1.5)),
        pelvis_floor_frac=float(detect_section.get("pelvis_floor_frac", 0.25)),
        recover_lift_rel=float(detect_section.get("recover_lift_rel", 0.25)),
        frame_skip=int(detect_section.get("frame_skip", 0)),
        candidate_alert_sec=float(detect_section.get("candidate_alert_sec", 10.0)),
        fallen_alert_sec=float(detect_section.get("fallen_alert_sec", 5.0)),
    )

    webhook = WebhookSection(
        url=str(webhook_section.get("url", "")),
        hmac_secret=webhook_section.get("hmac_secret"),
        timeout_seconds=float(webhook_section.get("timeout_seconds", 3.0)),
        verify_ssl=bool(webhook_section.get("verify_ssl", True)),
        cooldown_seconds=float(webhook_section.get("cooldown_seconds", 5.0)),
    )

    streams = [str(s) for s in streams_section]

    return Config(app=app, detect=detect, webhook=webhook, streams=streams)


__all__ = ["Config", "load_config"]
