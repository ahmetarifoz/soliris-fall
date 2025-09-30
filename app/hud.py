"""Heads-up display for fall detection tracks."""

import time
from typing import Dict, Any

import cv2

COL = {
    "bg": (24, 24, 24),
    "panel": (32, 32, 48),
    "text": (240, 240, 240),
    "muted": (180, 180, 180),
    "ok": (40, 180, 80),
    "warn": (0, 165, 255),
    "alert": (0, 0, 255),
    "badge": (60, 120, 240),
    "grid": (70, 70, 90),
}

STATE_RANK = {"fallen": 3, "candidate": 2, "recover": 1, "idle": 0}


def _put_text(img, msg, org, scale=0.6, color=(240, 240, 240), thick=2):
    cv2.putText(
        img, msg, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
    )


def _round_rect(img, x, y, w, h, r, color, fill=False, thick=2):
    if fill:
        overlay = img.copy()
        cv2.rectangle(overlay, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(overlay, (x, y + r), (x + w, y + h - r), color, -1)
        cv2.circle(overlay, (x + r, y + r), r, color, -1)
        cv2.circle(overlay, (x + w - r, y + r), r, color, -1)
        cv2.circle(overlay, (x + r, y + h - r), r, color, -1)
        cv2.circle(overlay, (x + w - r, y + h - r), r, color, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    else:
        cv2.line(img, (x + r, y), (x + w - r, y), color, thick)
        cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thick)
        cv2.line(img, (x, y + r), (x, y + h - r), color, thick)
        cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thick)
        cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thick)
        cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thick)
        cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thick)
        cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thick)


def _badge(img, x, y, text, color_bg, color_fg=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    pad = 8
    w, h = tw + pad * 2, th + pad
    _round_rect(img, x, y, w, h, 8, color_bg, fill=True)
    _put_text(img, text, (x + pad, y + h - pad // 2), 0.55, color_fg, 2)
    return x + w + 8, y + h


class HUD:
    """Render fall detection diagnostics onto video frames."""

    def __init__(self, max_tracks: int = 6):
        self.last_banner_t = 0.0
        self.max_tracks = max_tracks

    def draw_header(self, frame, any_alert):
        h, w = frame.shape[:2]
        _round_rect(frame, 10, 10, w - 20, 50, 12, COL["panel"], fill=True)
        _put_text(frame, "FALL MONITOR", (24, 44), 0.9, COL["text"], 2)

        x = w - 10
        legends = [
            ("FALL", COL["alert"]),
            ("RECOVER", COL["ok"]),
            ("CANDIDATE", COL["warn"]),
        ]
        for label, c in reversed(legends):
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            pad = 8
            width = tw + pad * 2
            x -= width + 10
            _round_rect(frame, x, 16, width, 28, 8, c, fill=True)
            _put_text(frame, label, (x + pad, 36), 0.55, (255, 255, 255), 2)

        if any_alert:
            self.last_banner_t = time.time()

        if time.time() - self.last_banner_t < 2.0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 10), COL["alert"], -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    def _order_and_clip(self, tracks_dict: Dict[int, Dict[str, Any]]):
        def pri(kv):
            tid, info = kv
            alert = 1 if info.get("alert", False) else 0
            state_rank = STATE_RANK.get(info.get("state", "idle"), 0)
            last_seen = float(info.get("last_seen", 0.0))
            return (alert, state_rank, last_seen)

        items = sorted(tracks_dict.items(), key=pri, reverse=True)[: self.max_tracks]
        return dict(items), {tid for tid, _ in items}

    def draw_track_card(self, frame, tid, info, idx):
        x, y = 12, 72 + idx * 88
        w, h = 360, 78
        _round_rect(frame, x, y, w, h, 10, COL["panel"], fill=True)

        state = info["state"].upper()
        color = {
            "FALLEN": COL["alert"],
            "CANDIDATE": COL["warn"],
            "RECOVER": COL["ok"],
            "IDLE": COL["muted"],
        }.get(state, COL["muted"])

        _put_text(frame, f"ID {tid}", (x + 14, y + 28), 0.7, COL["text"], 2)
        _badge(frame, x + 100, y + 8, state, color)

        _put_text(
            frame, f"ang {info['angle']:.1f}°", (x + 14, y + 52), 0.55, COL["muted"], 1
        )
        _put_text(
            frame, f"v {info['v']:.2f} bl/s", (x + 120, y + 52), 0.55, COL["muted"], 1
        )
        _put_text(
            frame, f"dr {info['drop']:.2f}", (x + 220, y + 52), 0.55, COL["muted"], 1
        )
        _put_text(
            frame, f"ar {info['ar']:.2f}", (x + 300, y + 52), 0.55, COL["muted"], 1
        )

        now = time.time()
        dur = min(3.0, now - info.get("since", now))
        bar_w = w - 20
        bar_x, bar_y = x + 10, y + h - 12
        cv2.line(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y), COL["grid"], 4)
        fill = int(bar_w * (dur / 3.0))
        cv2.line(frame, (bar_x, bar_y), (bar_x + fill, bar_y), color, 4)

        if info.get("alert", False):
            _badge(frame, x + w - 86, y + 8, "FALL", COL["alert"])

    def draw(self, frame, tracks_dict: Dict[int, Dict[str, Any]]):
        filtered, _ = self._order_and_clip(tracks_dict)

        any_alert = any(v.get("alert", False) for v in filtered.values())
        self.draw_header(frame, any_alert)

        for i, (tid, info) in enumerate(filtered.items()):
            self.draw_track_card(frame, tid, info, i)

        for tid, info in filtered.items():
            if "box" not in info or info["box"] is None:
                continue
            x1, y1, x2, y2 = map(int, info["box"])
            label = f"{tid}:{info['state'].upper()}"
            col = (
                COL["alert"]
                if info["state"] == "fallen"
                else (
                    COL["ok"]
                    if info["state"] == "recover"
                    else COL["warn"] if info["state"] == "candidate" else COL["badge"]
                )
            )
            _round_rect(frame, x1, max(0, y1 - 28), 120, 22, 6, col, fill=True)
            _put_text(
                frame, label, (x1 + 8, max(14, y1 - 10)), 0.55, (255, 255, 255), 2
            )


__all__ = ["HUD"]
