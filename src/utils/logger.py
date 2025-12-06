# src/utils/logger.py
import logging
from pathlib import Path
from typing import Optional

import yaml

from src.utils.paths import get_logging_config_path, get_logs_dir

_LOGGING_INITIALIZED = False


def load_logging_config(config_path: Path) -> Optional[dict]:
    """Load YAML config or return None if missing or invalid."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[WARN] Could not load logging config at {config_path}: {e}")
        return None


def setup_global_logger(config_path: Optional[Path] = None) -> None:
    """Initialize logging system with YAML config or fallback."""
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    config_path = config_path or get_logging_config_path()
    config = load_logging_config(config_path)

    logs_dir = get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)

    fmt = (
        config["format"]["ascii"]
        if config
        else "| {asctime} | {levelname:<8} | {name:<28} | {message}"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt, style="{"))
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(logs_dir / "main.log")
    file_handler.setFormatter(logging.Formatter(fmt, style="{"))
    file_handler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)

    root_logger.info("=" * 80)
    root_logger.info(f"Logging initialized from: {config_path}")
    root_logger.info(f"Logs directory: {logs_dir}")
    root_logger.info("=" * 80)

    _LOGGING_INITIALIZED = True


def get_global_logger(name: str) -> logging.Logger:
    """Return or initialize a named logger."""
    setup_global_logger()
    return logging.getLogger(name)
