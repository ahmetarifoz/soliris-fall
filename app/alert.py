"""Alert and OP500 trigger support for fall detection."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from typing import Optional

import requests

from . import logger


def _hmac_signature(secret: str, body: dict) -> str:
    msg = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


async def send_webhook(url: str, payload: dict, secret: str = "", max_retries: int = 3):
    """Send webhook with HMAC signature and retry logic."""
    if not url:
        return

    headers = {"Content-Type": "application/json"}
    if secret:
        headers["X-Signature"] = _hmac_signature(secret, payload)

    camera_id = payload.get("camera_id", "unknown")
    status = payload.get("status", "unknown")
    track_id = payload.get("track_id", 0)

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=10.0)
            r.raise_for_status()
            logger.webhook_sent(camera_id, status, track_id)
            return
        except Exception as e:
            logger.warning(f"Webhook attempt {attempt}/{max_retries} failed: {e}")
            await asyncio.sleep(backoff)
            backoff *= 2
    logger.webhook_failed(camera_id, "Max retries exceeded")


async def send_op500_trigger(base_url: str, port: int, event_type: str, delay_sec: float = 0):
    """Send OP500 trigger with optional delay."""
    if delay_sec > 0:
        logger.debug(f"Delaying OP500 trigger for {delay_sec}s...")
        await asyncio.sleep(delay_sec)

    url = f"{base_url.rstrip('/')}/trigger/{port}?type={event_type}"

    try:
        r = requests.post(url, json={}, timeout=30.0)
        if r.status_code == 200:
            logger.op500_trigger(port, event_type)
        else:
            logger.error(f"OP500 trigger failed ({r.status_code}): {r.text}")
    except Exception as e:
        logger.error(f"OP500 trigger error: {e}")


def send_op500_trigger_background(base_url: str, port: int, event_type: str, delay_sec: float = 0):
    """Launch OP500 trigger as a background task (non-blocking)."""
    asyncio.create_task(send_op500_trigger(base_url, port, event_type, delay_sec))


class RtpLoopbackManager:
    """Manages RTP loopback state and auto-close for fall detection."""

    def __init__(
        self,
        op500_enabled: bool = False,
        op500_base_url: Optional[str] = None,
        op500_rtp_port: Optional[int] = None,
        op500_rtp_auto_close_seconds: int = 120,
    ):
        self.op500_enabled = op500_enabled
        self.op500_base_url = op500_base_url
        self.op500_rtp_port = op500_rtp_port
        self.op500_rtp_auto_close_seconds = op500_rtp_auto_close_seconds

        self.rtp_loopback_active = False
        self.last_detection_time = 0.0
        self.stream_reader = None

    def set_stream_reader(self, stream_reader):
        """Set the stream reader reference for RTP control."""
        self.stream_reader = stream_reader

    def start_rtp_loopback(self) -> bool:
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

    def stop_rtp_loopback(self):
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
                print(f"RTP loopback stop failed: status={response.status_code}")
        except Exception as e:
            print(f"RTP loopback stop error: {e}")

    def check_rtp_auto_close(self):
        """Check if RTP loopback should auto-close due to no recent detections."""
        import time

        if not self.rtp_loopback_active:
            return

        elapsed = time.time() - self.last_detection_time

        if elapsed > self.op500_rtp_auto_close_seconds:
            print(f"RTP auto-close triggered: no detection for {elapsed:.0f}s (threshold={self.op500_rtp_auto_close_seconds}s)")
            self.stop_rtp_loopback()

    def update_detection_time(self):
        """Update last detection time to prevent auto-close."""
        import time
        self.last_detection_time = time.time()


__all__ = [
    "send_webhook",
    "send_op500_trigger",
    "send_op500_trigger_background",
    "RtpLoopbackManager",
]
