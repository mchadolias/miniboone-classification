from pathlib import Path
import os
from dotenv import load_dotenv
from typing import Optional


def get_project_root(start: Optional[Path] = None) -> Path:
    """
    Automatically locate the project root directory by walking upward
    until 'pyproject.toml' or '.git' is found.
    """
    start = start or Path(__file__).resolve()
    for parent in start.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def get_src_root() -> Path:
    """Return the absolute path to the `src/` directory."""
    return get_project_root() / "src"


def resolve_path_from_env(var_name: str, default: Path) -> Path:
    """
    Resolve a path from .env or use a default fallback.
    Expands '~' and environment variables.
    """
    load_dotenv()
    env_value = os.getenv(var_name)
    if env_value:
        return Path(os.path.expandvars(os.path.expanduser(env_value))).resolve()
    return default


def get_logging_config_path() -> Path:
    """
    Resolve the logging.yaml path using:
      1. LOGGING_CONFIG in .env (if exists)
      2. Default at src/config/logging.yaml
    """
    return resolve_path_from_env(
        "LOGGING_CONFIG",
        get_src_root() / "config" / "logging.yaml",
    )


def get_logs_dir() -> Path:
    """
    Resolve the logs directory path.
    Default: project_root / "logs"
    """
    return resolve_path_from_env("LOGS_DIR", get_project_root() / "logs")
