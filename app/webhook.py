"""Webhook notifications for fall detection events."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Dict, Optional

import requests

from .config import WebhookSection
from .detector import FallEvent


class WebhookClient:
    def __init__(self, cfg: WebhookSection):
        self.cfg = cfg
        self._last_sent: Dict[int, float] = {}

    def _should_send(self, track_id: int, now: float) -> bool:
        last = self._last_sent.get(track_id, 0.0)
        if now - last < self.cfg.cooldown_seconds:
            return False
        self._last_sent[track_id] = now
        return True

    def send(self, event: FallEvent) -> Optional[requests.Response]:
        if not self.cfg.url:
            return None
        now = float(event.timestamp)
        if not self._should_send(event.track_id, now):
            return None

        payload = event.to_payload()
        print(json.dumps(payload, indent=4))
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.cfg.hmac_secret:
            digest = hmac.new(
                self.cfg.hmac_secret.encode("utf-8"),
                body,
                hashlib.sha256,
            ).hexdigest()
            headers["X-Signature"] = digest

        try:
            response = requests.post(
                self.cfg.url,
                data=body,
                headers=headers,
                timeout=self.cfg.timeout_seconds,
                verify=self.cfg.verify_ssl,
            )
            response.raise_for_status()
            return response
        except requests.RequestException:
            return None


__all__ = ["WebhookClient"]
