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
class Op500Section:
    enabled: bool
    base_url: Optional[str]
    rtp_auto_close_seconds: int


@dataclass(frozen=True)
class ApiSection:
    """API server settings."""
    enabled: bool
    host: str
    port: int


@dataclass(frozen=True)
class FFmpegSection:
    """FFmpeg backend settings (like fsdapp's FFmpegCfg)."""
    hw: Optional[str]  # 'cuda' or None for CPU
    loglevel: str
    width: int
    height: int
    queue_size: int
    reconnect_seconds: float


@dataclass(frozen=True)
class IngestSection:
    """Ingest settings (like fsdapp's IngestCfg)."""
    backend: str  # 'ffmpeg' or 'opencv'
    ffmpeg: FFmpegSection


@dataclass(frozen=True)
class StreamSection:
    """Per-stream configuration (like fsdapp's StreamCfg)."""
    camera_id: str
    source: str
    backend: Optional[str]  # None = inherit from ingest
    width: Optional[int]  # None = inherit from ingest.ffmpeg
    height: Optional[int]
    ffmpeg_hw: Optional[str]  # None = inherit from ingest.ffmpeg
    queue_size: Optional[int]
    reconnect_seconds: Optional[float]
    rtp_port: Optional[int]  # RTP port for OP500 trigger mapping
    rtp_loopback_enabled: bool  # Enable RTP loopback


@dataclass(frozen=True)
class HudSection:
    """HUD settings (like fsdapp's HudCfg)."""
    fire_persistence_seconds: float
    alert_persistence_seconds: float


@dataclass(frozen=True)
class Config:
    app: AppSection
    detect: DetectSection
    webhook: WebhookSection
    op500: Optional[Op500Section]
    api: Optional[ApiSection]
    ingest: IngestSection
    hud: HudSection
    streams: List[StreamSection]  # Changed from List[str] to List[StreamSection] like fsdapp


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

    # OP500 configuration
    op500_section = data.get("op500", {})
    op500 = None
    if op500_section:
        op500 = Op500Section(
            enabled=bool(op500_section.get("enabled", False)),
            base_url=op500_section.get("base_url"),
            rtp_auto_close_seconds=int(op500_section.get("rtp_auto_close_seconds", 120)),
        )

    # Ingest configuration (like fsdapp)
    ingest_section = data.get("ingest", {})
    ffmpeg_section = ingest_section.get("ffmpeg", {})
    ffmpeg = FFmpegSection(
        hw=ffmpeg_section.get("hw", "cuda"),
        loglevel=str(ffmpeg_section.get("loglevel", "warning")),
        width=int(ffmpeg_section.get("width", 1920)),
        height=int(ffmpeg_section.get("height", 1080)),
        queue_size=int(ffmpeg_section.get("queue_size", 3)),
        reconnect_seconds=float(ffmpeg_section.get("reconnect_seconds", 2.0)),
    )
    ingest = IngestSection(
        backend=str(ingest_section.get("backend", "ffmpeg")),
        ffmpeg=ffmpeg,
    )

    # HUD configuration (like fsdapp)
    hud_section = data.get("hud", {})
    hud = HudSection(
        fire_persistence_seconds=float(hud_section.get("fire_persistence_seconds", 30.0)),
        alert_persistence_seconds=float(hud_section.get("alert_persistence_seconds", 30.0)),
    )

    # API server configuration
    api_section = data.get("api", {})
    api = None
    if api_section:
        api = ApiSection(
            enabled=bool(api_section.get("enabled", False)),
            host=str(api_section.get("host", "0.0.0.0")),
            port=int(api_section.get("port", 8000)),
        )

    # Stream configurations (like fsdapp's List[StreamCfg])
    streams = []
    for sc in streams_section:
        # Support both string format (legacy) and dict format (like fsdapp)
        if isinstance(sc, str):
            # Legacy format: just a source string
            streams.append(StreamSection(
                camera_id="",
                source=sc,
                backend=None,
                width=None,
                height=None,
                ffmpeg_hw=None,
                queue_size=None,
                reconnect_seconds=None,
                rtp_port=None,
                rtp_loopback_enabled=False,
            ))
        else:
            # Dict format like fsdapp
            streams.append(StreamSection(
                camera_id=str(sc.get("camera_id", "")),
                source=str(sc.get("source", "")),
                backend=sc.get("backend"),  # None = inherit
                width=sc.get("width"),  # None = inherit
                height=sc.get("height"),
                ffmpeg_hw=sc.get("ffmpeg_hw"),
                queue_size=sc.get("queue_size"),
                reconnect_seconds=sc.get("reconnect_seconds"),
                rtp_port=sc.get("rtp_port"),
                rtp_loopback_enabled=bool(sc.get("rtp_loopback_enabled", False)),
            ))

    return Config(app=app, detect=detect, webhook=webhook, op500=op500, api=api, ingest=ingest, hud=hud, streams=streams)


def resolve_stream_ingest(cfg: Config, s: StreamSection) -> Dict[str, Any]:
    """
    Resolve effective ingest settings for a single stream (like fsdapp's resolve_stream_ingest).
    Returns: {
      'backend': 'ffmpeg'|'opencv',
      'width': int,
      'height': int,
      'ffmpeg_hw': 'cuda'|None,
      'queue_size': int,
      'reconnect_seconds': float,
      'loglevel': str
    }
    """
    backend = s.backend or cfg.ingest.backend
    if backend == "ffmpeg":
        base = cfg.ingest.ffmpeg
        return {
            "backend": "ffmpeg",
            "width": s.width if s.width is not None else base.width,
            "height": s.height if s.height is not None else base.height,
            "ffmpeg_hw": s.ffmpeg_hw if s.ffmpeg_hw is not None else base.hw,
            "queue_size": s.queue_size if s.queue_size is not None else base.queue_size,
            "reconnect_seconds": s.reconnect_seconds if s.reconnect_seconds is not None else base.reconnect_seconds,
            "loglevel": base.loglevel,
        }
    else:
        # OpenCV backend - width/height still useful
        base = cfg.ingest.ffmpeg
        return {
            "backend": "opencv",
            "width": s.width if s.width is not None else base.width,
            "height": s.height if s.height is not None else base.height,
            "ffmpeg_hw": None,
            "queue_size": s.queue_size if s.queue_size is not None else base.queue_size,
            "reconnect_seconds": s.reconnect_seconds if s.reconnect_seconds is not None else base.reconnect_seconds,
            "loglevel": base.loglevel,
        }


__all__ = ["Config", "load_config", "resolve_stream_ingest", "StreamSection"]
