"""Centralized logging for fall detection system."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# ANSI color codes
COLORS = {
    "RESET": "\033[0m",
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "YELLOW": "\033[93m",
    "BLUE": "\033[94m",
    "MAGENTA": "\033[95m",
    "CYAN": "\033[96m",
    "WHITE": "\033[97m",
    "BOLD": "\033[1m",
}


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    LEVEL_COLORS = {
        logging.DEBUG: COLORS["CYAN"],
        logging.INFO: COLORS["GREEN"],
        logging.WARNING: COLORS["YELLOW"],
        logging.ERROR: COLORS["RED"],
        logging.CRITICAL: COLORS["MAGENTA"] + COLORS["BOLD"],
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, COLORS["WHITE"])
        reset = COLORS["RESET"]
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        
        # Build message
        level = record.levelname[:4]
        msg = f"{color}[{timestamp}] [{level}]{reset} {record.getMessage()}"
        
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        
        return msg


class FileFormatter(logging.Formatter):
    """Plain formatter for file output."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level = record.levelname
        msg = f"[{timestamp}] [{level}] {record.name}: {record.getMessage()}"
        
        if record.exc_info:
            msg += f"\n{self.formatException(record.exc_info)}"
        
        return msg


def setup_logger(
    name: str = "fall",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup and return a configured logger."""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(FileFormatter())
        logger.addHandler(file_handler)
    
    return logger


# Default logger instance
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get the default logger instance."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


# Convenience functions
def debug(msg: str, *args, **kwargs):
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs):
    get_logger().critical(msg, *args, **kwargs)


# Alert-specific logging
def alert_candidate(track_id: int, camera_id: str, angle: float, confidence: float):
    """Log a fall candidate detection."""
    get_logger().warning(
        f"⚠️  CANDIDATE | track={track_id} cam={camera_id} angle={angle:.1f}° conf={confidence:.2f}"
    )


def alert_fallen(track_id: int, camera_id: str, angle: float, confidence: float):
    """Log a confirmed fall detection."""
    get_logger().critical(
        f"🚨 FALL DETECTED | track={track_id} cam={camera_id} angle={angle:.1f}° conf={confidence:.2f}"
    )


def alert_recovered(track_id: int, camera_id: str):
    """Log a recovery from fall."""
    get_logger().info(
        f"✅ RECOVERED | track={track_id} cam={camera_id}"
    )


def alert_cooldown(track_id: int, remaining_sec: float):
    """Log cooldown status."""
    get_logger().debug(
        f"⏳ COOLDOWN | track={track_id} remaining={remaining_sec:.1f}s"
    )


def rtp_started(port: int, camera_id: str):
    """Log RTP stream start."""
    get_logger().info(
        f"📡 RTP START | port={port} cam={camera_id}"
    )


def rtp_stopped(port: int, camera_id: str):
    """Log RTP stream stop."""
    get_logger().info(
        f"📡 RTP STOP | port={port} cam={camera_id}"
    )


def webhook_sent(camera_id: str, status: str, track_id: int):
    """Log webhook sent."""
    get_logger().info(
        f"📤 WEBHOOK | cam={camera_id} status={status} track={track_id}"
    )


def webhook_failed(camera_id: str, error: str):
    """Log webhook failure."""
    get_logger().error(
        f"❌ WEBHOOK FAILED | cam={camera_id} error={error}"
    )


def op500_trigger(port: int, event_type: str):
    """Log OP500 trigger."""
    get_logger().info(
        f"🔔 OP500 TRIGGER | port={port} type={event_type}"
    )


def stream_connected(source: str):
    """Log stream connection."""
    get_logger().info(
        f"🎥 STREAM CONNECTED | {source}"
    )


def stream_disconnected(source: str):
    """Log stream disconnection."""
    get_logger().warning(
        f"🎥 STREAM DISCONNECTED | {source}"
    )


def stream_reconnecting(source: str, attempt: int):
    """Log stream reconnection attempt."""
    get_logger().warning(
        f"🔄 RECONNECTING | {source} attempt={attempt}"
    )
