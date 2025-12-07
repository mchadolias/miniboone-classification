import json
import joblib
from pathlib import Path
from datetime import datetime


def save_model(model, config: dict, metrics: dict, save_dir: Path = Path("models")):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = save_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model.pkl")

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return run_dir


def load_model(run_dir: Path, model_filename: str = "model.pkl"):
    model_path = run_dir / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Model file '{model_path}' does not exist.")

    model = joblib.load(model_path)
    return model
