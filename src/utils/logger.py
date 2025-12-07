import logging
import os
from pathlib import Path
from typing import Optional

from src.utils.paths import get_logs_dir

_LOGGING_INITIALIZED = False
_CURRENT_PROFILE: Optional[str] = None


def _setup_cli_logging(root_logger: logging.Logger) -> None:
    """Profile for scripts/CLIs: console + file."""
    fmt = "| {asctime} | {levelname:<8} | {name:<28} | {message}"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt, style="{", datefmt=datefmt))
    stream_handler.setLevel(logging.INFO)

    # File handler
    logs_dir: Path = get_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(logs_dir / "main.log")
    file_handler.setFormatter(logging.Formatter(fmt, style="{", datefmt=datefmt))
    file_handler.setLevel(logging.DEBUG)

    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)


def _setup_notebook_logging(root_logger: logging.Logger) -> None:
    """Profile for notebooks: simple console-only logging."""
    fmt = "[{levelname:<7}] {name}: {message}"
    datefmt = "%H:%M:%S"

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(fmt, style="{", datefmt=datefmt))
    stream_handler.setLevel(logging.INFO)

    root_logger.addHandler(stream_handler)


def setup_global_logger(
    profile: Optional[str] = None,
    *,
    force: bool = False,
) -> None:
    """
    Initialise root logging with a given profile.

    Parameters
    ----------
    profile : {'cli', 'notebook', None}
        - 'cli'      → console + file logger (for scripts/MLflow runs) [DEFAULT]
        - 'notebook' → console only, compact format
        - None       → use MINIBOONE_LOG_PROFILE or 'cli' if unset
    force : bool
        If True, reconfigure even if already initialised.
    """
    global _LOGGING_INITIALIZED, _CURRENT_PROFILE

    if _LOGGING_INITIALIZED and not force:
        return

    # env var > explicit arg > default 'cli'
    env_profile = os.environ.get("MINIBOONE_LOG_PROFILE")
    if profile is None:
        profile = env_profile or "cli"

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG)

    if profile == "notebook":
        _setup_notebook_logging(root_logger)
    else:
        _setup_cli_logging(root_logger)
        if profile not in ("cli", "notebook"):
            root_logger.warning("Unknown logging profile '%s'; falling back to 'cli'.", profile)
            profile = "cli"

    root_logger.info("=" * 80)
    root_logger.info("Logging initialised (profile=%s)", profile)
    root_logger.info("=" * 80)

    _LOGGING_INITIALIZED = True
    _CURRENT_PROFILE = profile


def get_global_logger(name: str, profile: Optional[str] = None) -> logging.Logger:
    """
    Get a named logger. Initialises global logging on first use.

    Parameters
    ----------
    name : str
        Logger name (usually __name__).
    profile : Optional[str]
        Optional profile override ('cli' or 'notebook').
        If None, uses setup_global_logger default logic.
    """
    setup_global_logger(profile=profile)
    return logging.getLogger(name)
