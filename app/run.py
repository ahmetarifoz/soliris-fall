"""Entrypoint for the fall detection runtime."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2

from .config import AppSection, Config, load_config
from .detector import FallDetector
from .hud import HUD
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


def _build_ffmpeg_options(app_cfg: AppSection) -> Optional[str]:
    opts = dict(app_cfg.rtsp_ffmpeg_options)
    if app_cfg.rtsp_transport and "rtsp_transport" not in opts:
        opts["rtsp_transport"] = app_cfg.rtsp_transport
    if not opts:
        return None
    return "|".join(f"{key}={value}" for key, value in opts.items())


def _create_capture(
    source: Any,
    backend: str,
    *,
    is_rtsp: bool,
    ffmpeg_options: Optional[str],
) -> cv2.VideoCapture:
    previous_env = None
    applied_options = False
    use_ffmpeg = backend.lower() == "ffmpeg"
    if is_rtsp and use_ffmpeg and ffmpeg_options:
        previous_env = os.environ.get(ENV_CAPTURE_OPTIONS)
        os.environ[ENV_CAPTURE_OPTIONS] = ffmpeg_options
        applied_options = True
    try:
        cap = (
            cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if is_rtsp and use_ffmpeg
            else cv2.VideoCapture(source)
        )
    finally:
        if applied_options:
            if previous_env is None:
                os.environ.pop(ENV_CAPTURE_OPTIONS, None)
            else:
                os.environ[ENV_CAPTURE_OPTIONS] = previous_env
    if is_rtsp and cap.isOpened():
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


def run(config_path: str | Path = "config.yaml") -> None:
    cfg = load_config(config_path)

    source_spec = cfg.streams[0]
    source = _resolve_source(source_spec)
    is_rtsp = _is_rtsp_source(source)
    ffmpeg_options = _build_ffmpeg_options(cfg.app)

    cap = _create_capture(
        source,
        cfg.app.rtsp_backend,
        is_rtsp=is_rtsp,
        ffmpeg_options=ffmpeg_options,
    )
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source_spec}")

    writer, fps = _ensure_writer(cfg, cap)

    detector = FallDetector(cfg, camera_id=str(source_spec))
    hud = HUD(max_tracks=cfg.app.max_tracks) if cfg.app.enable_hud else None
    webhook = WebhookClient(cfg.webhook)

    frame_skip = max(0, cfg.detect.frame_skip)
    skip_cursor = 0

    if hud:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print("Running. Press ESC to quit.")

    try:
        while True:
            ok, frame = _read_frame(
                cap,
                drop_frames=cfg.app.rtsp_drop_frames,
                is_rtsp=is_rtsp,
            )
            if not ok or frame is None:
                if is_rtsp and cfg.app.rtsp_reconnect:
                    print("Frame read failed; attempting RTSP reconnect...")
                    cap.release()
                    reconnected = False
                    for attempt in range(1, cfg.app.rtsp_max_retries + 1):
                        time.sleep(cfg.app.rtsp_retry_delay)
                        cap = _create_capture(
                            source,
                            cfg.app.rtsp_backend,
                            is_rtsp=True,
                            ffmpeg_options=ffmpeg_options,
                        )
                        if cap.isOpened():
                            fps = cap.get(cv2.CAP_PROP_FPS) or cfg.app.default_fps
                            if writer is None and cfg.app.save_path:
                                writer, fps = _ensure_writer(cfg, cap)
                            reconnected = True
                            skip_cursor = 0
                            print(f"RTSP stream reconnected on attempt {attempt}.")
                            break
                        else:
                            print(f"RTSP reconnect attempt {attempt} failed.")
                    if not reconnected:
                        print("RTSP reconnect attempts exhausted; exiting loop.")
                        break
                    continue
                print("Frame could not be read; exiting loop.")
                break

            if frame_skip > 0:
                skip_cursor = (skip_cursor + 1) % (frame_skip + 1)
                if skip_cursor:
                    continue

            result = detector.process_frame(frame, fps)

            for event in result.events:
                webhook.send(event)

            if hud:
                hud.draw(result.frame, result.tracks_view)
                cv2.imshow(WINDOW_NAME, result.frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if writer is not None:
                writer.write(result.frame)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if hud:
            cv2.destroyWindow(WINDOW_NAME)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
