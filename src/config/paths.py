from pathlib import Path


def get_project_root() -> Path:
    """Return absolute path to project root (one level above 'src')."""
    return Path(__file__).resolve().parents[1].parent


if __name__ == "__main__":
    print(f"Project root path: {get_project_root()}")
