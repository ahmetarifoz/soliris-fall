"""Deprecated module; prefer using app.config.load_config."""

from app.config import load_config, Config

__all__ = ["load_config", "Config"]
