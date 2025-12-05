import logging
import sys
import os
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any


# --- Colors ---
class LogColors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    GRAY = "\033[90m"


def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


# --- Formatters ---
class ASCIILogFormatter(logging.Formatter):
    def format(self, record):
        return f"| {self.formatTime(record, '%Y-%m-%d %H:%M:%S')} | {record.levelname:<8} | {record.name:<25} | {record.getMessage()}"


class ColorLogFormatter(logging.Formatter):
    LEVEL_COLORS = {
        "DEBUG": LogColors.GRAY + LogColors.DIM,
        "INFO": LogColors.GREEN,
        "WARNING": LogColors.YELLOW,
        "ERROR": LogColors.RED + LogColors.BOLD,
        "CRITICAL": LogColors.MAGENTA + LogColors.BOLD,
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelname, LogColors.RESET)
        reset = LogColors.RESET
        return f"{color}| {self.formatTime(record, '%H:%M:%S')} | {record.levelname:<8} | {record.name:<25} | {record.getMessage()}{reset}"


class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        obj = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(obj)


# --- YAML Configuration Loader ---
def load_logging_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML logging configuration."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# --- Logger Factory ---
def get_logger_from_config(
    name: str,
    config_path: Optional[Path] = None,
) -> logging.Logger:
    """
    Create a logger from YAML configuration file.
    Automatically resolves the project root and falls back to defaults if missing.
    """
    # --- Resolve config path automatically ---
    if config_path is None or not Path(config_path).exists():
        # Locate the project root relative to this file (src/utils/logger.py)
        base_dir = Path(__file__).resolve().parents[1]  # points to src/
        candidate_path = base_dir / "config/logging.yaml"

        if candidate_path.exists():
            config_path = candidate_path
        else:
            print(
                f"[WARNING] Logging config not found at {candidate_path}, using fallback config."
            )
            # fallback minimal configuration
            config = {
                "defaults": {
                    "console_level": "INFO",
                    "file_level": "DEBUG",
                    "json_logging": False,
                    "enable_color": True,
                    "log_dir": "logs",
                }
            }
            return _create_logger_from_dict(name, config)

    # Load YAML config if file exists
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return _create_logger_from_dict(name, config)


def _create_logger_from_dict(name: str, config: dict) -> logging.Logger:
    """Helper to instantiate a logger from config dict."""
    defaults = config.get("defaults", {})
    log_dir = Path(defaults.get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    console_level = getattr(logging, defaults.get("console_level", "INFO"))
    file_level = getattr(logging, defaults.get("file_level", "DEBUG"))
    enable_color = defaults.get("enable_color", True)
    json_logging = defaults.get("json_logging", False)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(min(console_level, file_level))

    # --- Console handler ---
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(console_level)
    console.setFormatter(
        ColorLogFormatter() if enable_color and supports_color() else ASCIILogFormatter()
    )
    logger.addHandler(console)

    # --- File handler ---
    file_handler = logging.FileHandler(
        log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{name.replace('.', '_')}.log"
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(ASCIILogFormatter())
    logger.addHandler(file_handler)

    logger.propagate = False
    logger.debug(f"Logger initialized for {name}")
    return logger
