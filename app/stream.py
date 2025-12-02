"""Stream reader with RTP loopback support for fall detection."""

from __future__ import annotations

import logging
import os
import platform
import queue
import subprocess
import threading
import time
from collections import deque
from typing import Generator, Optional

import cv2
import numpy as np
import requests

# Use module-level logger to avoid circular imports
_log = logging.getLogger("fall.stream")


def _ensure_rtsp_over_tcp():
    key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    base_opts = [
        "rtsp_transport;tcp",
        "rtsp_flags;prefer_tcp",
        "max_delay;500000",
        "probesize;32",
        "analyzeduration;0",
        "fflags;nobuffer",
        "fflags;discardcorrupt",
        "flags;low_delay",
        "err_detect;ignore_err",
    ]
    wanted = "|".join(base_opts)
    cur = os.environ.get(key)
    if not cur:
        os.environ[key] = wanted
        return
    have = {kv.split(";")[0] for kv in cur.split("|") if kv}
    extra = [kv for kv in base_opts if kv.split(";")[0] not in have]
    if extra:
        os.environ[key] = cur.rstrip("|") + "|" + "|".join(extra)


def _maybe_enable_hw_accel(cap: cv2.VideoCapture, device_index: int) -> bool:
    if not hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
        return False
    accel_modes = []
    for attr in ("VIDEO_ACCELERATION_D3D11", "VIDEO_ACCELERATION_ANY"):
        value = getattr(cv2, attr, None)
        if value is not None:
            accel_modes.append((attr, value))
    for name, value in accel_modes:
        try:
            if not cap.set(cv2.CAP_PROP_HW_ACCELERATION, value):
                continue
            current = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION))
            if current != int(value):
                continue
            if hasattr(cv2, "CAP_PROP_HW_DEVICE"):
                cap.set(cv2.CAP_PROP_HW_DEVICE, device_index)
            _log.info(f"Hardware video decode via {name.replace('VIDEO_ACCELERATION_', '')} enabled")
            return True
        except cv2.error:
            pass
    return False


def _open_capture(source: str, prefer_hw: bool, hw_device_index: int):
    """Open video capture with optional hardware acceleration."""
    if source.startswith("webcam:") or source.isdigit():
        index = int(source.split(":")[1]) if ":" in source else int(source)
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
        cap = cv2.VideoCapture(index, backend)
        return cap, False
    if os.path.isfile(source):
        cap = cv2.VideoCapture(source)
        return cap, False
    if source.startswith("rtsp://"):
        _ensure_rtsp_over_tcp()
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    else:
        cap = cv2.VideoCapture(source)

    using_hw = False
    if cap and cap.isOpened() and prefer_hw:
        using_hw = _maybe_enable_hw_accel(cap, hw_device_index)
    return cap, using_hw


class StreamReader:
    """Video stream reader with RTP loopback support."""

    def __init__(
        self,
        source: str,
        reconnect_delay: float = 1.0,
        loop_file: bool = True,
        prefer_hw_accel: bool = True,
        hw_device_index: int = 0,
        width: int = 1920,
        height: int = 1080,
        backend: str = "opencv",
        ffmpeg_hw: Optional[str] = "cuda",
        ffmpeg_loglevel: str = "warning",
        queue_size: int = 2,
        ffmpeg_bin: Optional[str] = None,
        rtp_output_host: Optional[str] = None,
        rtp_output_port: Optional[int] = None,
    ):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.loop_file = loop_file
        self.prefer_hw_accel = prefer_hw_accel
        self.hw_device_index = hw_device_index
        self._is_file = os.path.isfile(source)

        # OpenCV
        self.cap: Optional[cv2.VideoCapture] = None
        self._using_hw_accel = False

        # FFmpeg
        self.backend = backend
        self.width = width
        self.height = height
        self.ffmpeg_hw = ffmpeg_hw
        self.ffmpeg_loglevel = ffmpeg_loglevel
        self.ffmpeg_bin = ffmpeg_bin or os.environ.get("FFMPEG_BIN", "ffmpeg")
        self.proc: Optional[subprocess.Popen] = None
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._frame_bytes = self.width * self.height * 3
        self._stderr_tail: deque = deque(maxlen=80)
        self._use_scale_cuda = False

        # RTP output
        self._parse_rtp_info_from_source()
        self.rtp_output_host = rtp_output_host or self._detected_host or "127.0.0.1"
        self.rtp_output_port = rtp_output_port
        self._rtp_proc: Optional[subprocess.Popen] = None
        self._rtp_enabled = rtp_output_port is not None
        self._rtp_active = False
        self._rtp_last_frame_time = 0.0
        self._rtp_stream_duration = 120.0  # Stream for 120 seconds after detection
        self._rtp_reconnect_attempts = 0
        self._rtp_max_reconnect_attempts = 30
        self._rtp_reconnect_delay = 3.0
        self._rtp_last_reconnect_time = 0.0
        self._rtp_last_successful_send = 0.0
        self._rtp_health_check_interval = 5.0
        self._rtp_freeze_timeout = 10.0
        self._rtp_registered = False

        # OP500 integration
        self.op500_base_url: Optional[str] = None
        self.op500_enabled = False

    def _parse_rtp_info_from_source(self):
        """Parse RTP host/port from SDP file or UDP source."""
        self._detected_host = None
        self._detected_port = None
        self._detected_payload_type = 96

        if self.source.lower().endswith(".sdp") or "sdp://" in self.source.lower():
            sdp_path = self.source.replace("sdp://", "")
            try:
                with open(sdp_path, "r") as f:
                    for line in f:
                        if line.startswith("c=IN IP4"):
                            parts = line.strip().split()
                            if len(parts) >= 3:
                                self._detected_host = parts[2]
                        elif line.startswith("m=video"):
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                try:
                                    self._detected_port = int(parts[1])
                                except ValueError:
                                    pass
                            if len(parts) >= 4:
                                try:
                                    self._detected_payload_type = int(parts[3])
                                except ValueError:
                                    pass
            except Exception:
                pass
        elif "udp://" in self.source.lower():
            import re
            match = re.search(r"udp://([^:]+):(\d+)", self.source)
            if match:
                self._detected_host = match.group(1)
                self._detected_port = int(match.group(2))

    def _has_filter(self, name: str) -> bool:
        try:
            out = subprocess.check_output(
                [self.ffmpeg_bin, "-hide_banner", "-filters"],
                stderr=subprocess.STDOUT,
                text=True,
            )
            return name in out
        except Exception:
            return False

    def _drain_stderr(self, proc: subprocess.Popen):
        for line in iter(proc.stderr.readline, b""):
            try:
                s = line.decode(errors="ignore").rstrip()
            except Exception:
                s = str(line)
            if s:
                self._stderr_tail.append(s)

    def _ffmpeg_cmd(self):
        use_scale_npp = self._has_filter("scale_npp")
        use_scale_cuda = (self.ffmpeg_hw == "cuda") and self._has_filter("scale_cuda")

        if self.ffmpeg_hw == "cuda" and use_scale_npp:
            vf = (
                f"scale_npp={self.width}:{self.height}:interp_algo=super,"
                f"hwdownload,format=nv12,format=bgr24"
            )
        elif self.ffmpeg_hw == "cuda" and use_scale_cuda:
            vf = (
                f"scale_cuda={self.width}:{self.height}:interp_algo=lanczos,"
                f"hwdownload,format=nv12,format=bgr24"
            )
        elif self.ffmpeg_hw == "cuda":
            vf = (
                f"hwdownload,format=nv12,"
                f"scale={self.width}:{self.height}:flags=lanczos,format=bgr24"
            )
        else:
            vf = f"scale={self.width}:{self.height}:flags=lanczos,format=bgr24"

        src = self.source
        is_rtsp = isinstance(src, str) and src.startswith("rtsp://")
        is_sdp_scheme = isinstance(src, str) and src.startswith("sdp://")
        is_sdp_file = isinstance(src, str) and src.lower().endswith(".sdp")
        sdp_path = src[6:] if is_sdp_scheme else src

        common = [
            self.ffmpeg_bin,
            "-hide_banner",
            "-loglevel", self.ffmpeg_loglevel,
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-hwaccel_device", "0",
            "-extra_hw_frames", "10",
            "-c:v", "h264_cuvid",
            "-fflags", "+nobuffer+fastseek+flush_packets",
            "-err_detect", "ignore_err",
            "-flags", "low_delay",
            "-strict", "experimental",
            "-probesize", "100M",
            "-analyzeduration", "10000000",
            "-thread_queue_size", "512",
        ]

        args = list(common)

        if is_rtsp:
            args += ["-rtsp_transport", "tcp", "-rtsp_flags", "prefer_tcp", "-i", src]
        elif is_sdp_scheme or is_sdp_file:
            args += ["-protocol_whitelist", "file,udp,rtp,tcp,crypto,data", "-i", sdp_path]
        else:
            args += ["-re", "-i", src]

        args += ["-an", "-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
        return args

    def _ffmpeg_reader(self):
        while not self._stop.is_set():
            if self.ffmpeg_hw == "cuda" and not self._use_scale_cuda:
                self._use_scale_cuda = self._has_filter("scale_cuda")

            try:
                self.proc = subprocess.Popen(
                    self._ffmpeg_cmd(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=10**7,
                )
            except FileNotFoundError:
                _log.error("ffmpeg not found. Add to PATH or set ffmpeg_bin parameter.")
                return

            err_thr = threading.Thread(
                target=self._drain_stderr, args=(self.proc,), daemon=True
            )
            err_thr.start()

            stdout = self.proc.stdout
            acc = b""
            rc = 0
            while not self._stop.is_set():
                if self.proc.poll() is not None:
                    rc = self.proc.returncode
                    break

                need = self._frame_bytes - len(acc)
                chunk = stdout.read(need)
                if not chunk:
                    time.sleep(0.02)
                    continue

                acc += chunk
                if len(acc) == self._frame_bytes:
                    frame = np.frombuffer(acc, dtype=np.uint8).reshape(
                        self.height, self.width, 3
                    ).copy()
                    acc = b""
                    if self._q.full():
                        try:
                            self._q.get_nowait()
                        except queue.Empty:
                            pass
                    self._q.put(frame)

            try:
                self.proc.kill()
            except Exception:
                pass

            if self._is_file and not self.loop_file and rc == 0:
                break

            time.sleep(self.reconnect_delay)

    def _start_ffmpeg(self):
        if self._t and self._t.is_alive():
            return True
        self._stop.clear()
        self._t = threading.Thread(target=self._ffmpeg_reader, daemon=True)
        self._t.start()
        return True

    def _open_opencv(self) -> bool:
        self.cap, self._using_hw_accel = _open_capture(
            self.source, self.prefer_hw_accel, self.hw_device_index
        )
        if not self.cap or not self.cap.isOpened():
            _log.error(f"Failed to open video source: {self.source}")
            return False
        return True

    def _open(self) -> bool:
        if self.backend == "ffmpeg":
            return self._start_ffmpeg()
        return self._open_opencv()

    def frames(self) -> Generator[Optional[np.ndarray], None, None]:
        if not self._open():
            yield None
            return

        if self.backend == "ffmpeg":
            while True:
                try:
                    frame = self._q.get(timeout=2)
                    yield frame
                except queue.Empty:
                    if self.proc and self.proc.poll() is not None:
                        _log.warning("FFmpeg process not running; waiting for reconnect...")
                    continue

        bad_reads = 0
        while self.backend == "opencv":
            ok, frame = self.cap.read()
            if not ok or frame is None:
                bad_reads += 1
                if self._is_file and self.loop_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    bad_reads = 0
                    continue
                if bad_reads <= 3:
                    time.sleep(0.05)
                    continue
                _log.warning(f"Frame read failed {bad_reads}x. Reopening {self.source}...")
                bad_reads = 0
                try:
                    self.cap.release()
                except Exception:
                    pass
                time.sleep(self.reconnect_delay)
                if not self._open_opencv():
                    time.sleep(self.reconnect_delay)
                try:
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                continue

            bad_reads = 0
            if getattr(frame, "ndim", 0) == 3:
                yield frame

    def _register_sender_with_op500(self) -> bool:
        """Register RTP sender with OP500 before starting stream."""
        if not self.op500_enabled or not self.op500_base_url:
            return True

        try:
            url = f"{self.op500_base_url.rstrip('/')}/detectors/sender/add"
            payload = {
                "host": self.rtp_output_host,
                "port": self.rtp_output_port,
                "codec": "h264",
                "payload_type": getattr(self, '_detected_payload_type', 96)
            }

            _log.info(f"Registering RTP sender with OP500: {url}")
            response = requests.post(url, json=payload, timeout=5)

            if response.status_code in (200, 201):
                _log.info("OP500 sender registration successful")
                return True
            else:
                _log.error(f"OP500 sender registration failed: {response.status_code}")
                return False

        except Exception as e:
            _log.error(f"Failed to register sender with OP500: {e}")
            return False

    def _start_rtp_sender(self):
        """Start FFmpeg process to send frames via RTP."""
        if not self._rtp_enabled:
            return

        if self._rtp_active and self._rtp_proc and self._rtp_proc.poll() is None:
            return

        if self._rtp_proc:
            try:
                self._rtp_proc.kill()
            except Exception:
                pass
            self._rtp_proc = None
            self._rtp_active = False

        if not self._rtp_registered:
            if self._register_sender_with_op500():
                self._rtp_registered = True

        try:
            rtp_url = f"rtp://{self.rtp_output_host}:{self.rtp_output_port}"
            payload_type = getattr(self, '_detected_payload_type', 96)

            cmd = [
                self.ffmpeg_bin,
                "-hide_banner",
                "-loglevel", "warning",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-framerate", "25",
                "-thread_queue_size", "512",
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-threads", "4",
                "-g", "50",
                "-bf", "0",
                "-profile:v", "baseline",
                "-level", "4.2",
                "-crf", "28",
                "-maxrate", "4M",
                "-bufsize", "2M",
                "-pix_fmt", "yuv420p",
                "-an",
                "-f", "rtp",
                "-payload_type", str(payload_type),
                rtp_url
            ]

            self._rtp_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**7
            )

            def read_stderr():
                for line in iter(self._rtp_proc.stderr.readline, b''):
                    try:
                        msg = line.decode('utf-8', errors='ignore').strip()
                        if msg:
                            _log.debug(f"RTP FFmpeg: {msg}")
                    except Exception:
                        pass

            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            self._rtp_active = True
            self._rtp_last_frame_time = time.time()
            self._rtp_last_successful_send = time.time()
            self._rtp_reconnect_attempts = 0
            _log.info(f"RTP sender started: {rtp_url}")
        except Exception as e:
            _log.error(f"Failed to start RTP sender: {e}")
            self._rtp_proc = None
            self._rtp_active = False

    def _stop_rtp_sender(self):
        """Stop the RTP sender process."""
        if self._rtp_proc:
            try:
                self._rtp_proc.stdin.close()
                self._rtp_proc.terminate()
                self._rtp_proc.wait(timeout=1)
            except Exception:
                try:
                    self._rtp_proc.kill()
                except Exception:
                    pass
            finally:
                self._rtp_proc = None
                self._rtp_active = False
                self._rtp_registered = False
                _log.info("RTP sender stopped")

    def trigger_rtp_stream(self):
        """Trigger RTP streaming for a duration after detection."""
        if not self._rtp_enabled:
            return

        if not self._rtp_active or not self._rtp_proc:
            self._start_rtp_sender()

        self._rtp_last_frame_time = time.time()
        if self._rtp_active and self._rtp_proc:
            self._rtp_reconnect_attempts = 0
            _log.debug(f"RTP stream triggered (duration: {self._rtp_stream_duration}s)")

    def _check_rtp_health(self):
        """Check RTP stream health - restart if frozen."""
        if not self._rtp_active or not self._rtp_proc:
            return

        if self._rtp_last_successful_send == 0.0:
            return

        current_time = time.time()
        time_since_last_send = current_time - self._rtp_last_successful_send

        if time_since_last_send > self._rtp_freeze_timeout:
            _log.warning(f"RTP stream frozen: no frames sent for {time_since_last_send:.1f}s. Restarting...")
            self._stop_rtp_sender()
            self._rtp_active = False

            if self._rtp_reconnect_attempts < self._rtp_max_reconnect_attempts:
                self._rtp_reconnect_attempts += 1
                self._rtp_last_reconnect_time = current_time
                self._start_rtp_sender()

    def send_frame_to_rtp(self, frame: np.ndarray):
        """Send a single frame to RTP output if streaming is active."""
        if not self._rtp_enabled or not self._rtp_active:
            return

        current_time = time.time()
        if self._rtp_last_successful_send > 0:
            if current_time - self._rtp_last_successful_send > self._rtp_health_check_interval:
                self._check_rtp_health()

        if current_time - self._rtp_last_frame_time > self._rtp_stream_duration:
            _log.info("RTP stream timeout, stopping sender")
            self._stop_rtp_sender()
            return

        if not self._rtp_proc:
            return

        try:
            if self._rtp_proc.poll() is not None:
                rc = self._rtp_proc.returncode
                _log.warning(f"RTP sender died with code {rc}. Attempting reconnect...")
                self._rtp_active = False
                self._rtp_proc = None

                if self._rtp_reconnect_attempts < self._rtp_max_reconnect_attempts:
                    backoff_delay = self._rtp_reconnect_delay * (2 ** self._rtp_reconnect_attempts)
                    if current_time - self._rtp_last_reconnect_time >= backoff_delay:
                        self._rtp_reconnect_attempts += 1
                        self._rtp_last_reconnect_time = current_time
                        self._start_rtp_sender()
                return

            if frame is None or frame.size == 0:
                return

            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                return

            frame_bytes = frame.tobytes()
            self._rtp_proc.stdin.write(frame_bytes)
            self._rtp_proc.stdin.flush()
            self._rtp_last_successful_send = time.time()

        except BrokenPipeError:
            _log.warning("RTP pipe broken, attempting reconnect...")
            self._rtp_proc = None
            self._rtp_active = False

            if self._rtp_reconnect_attempts < self._rtp_max_reconnect_attempts:
                backoff_delay = self._rtp_reconnect_delay * (2 ** self._rtp_reconnect_attempts)
                if current_time - self._rtp_last_reconnect_time >= backoff_delay:
                    self._rtp_reconnect_attempts += 1
                    self._rtp_last_reconnect_time = current_time
                    self._start_rtp_sender()
        except Exception as e:
            _log.error(f"Error sending frame to RTP: {e}")
            self._rtp_active = False

    def close(self):
        if self.backend == "ffmpeg":
            self._stop.set()
            if self.proc:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=1)
                except Exception:
                    pass
        else:
            try:
                if self.cap:
                    self.cap.release()
            except Exception:
                pass

        if self._rtp_proc:
            self._stop_rtp_sender()


__all__ = ["StreamReader"]
