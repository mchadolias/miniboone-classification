from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Any, Dict

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings


@dataclass
class SaveConfig:
    save_dir: Path = Path("output/")
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 300
    bbox_inches: str = "tight"
    transparent: bool = False

    def __post_init__(self) -> None:
        """Normalize save_dir to Path for consistent path handling."""
        self.save_dir = Path(self.save_dir)


@dataclass
class MLConfig:
    experiment_name: str = "miniboone_classification"
    run_name: str = "baseline_run"

    model_name: str = "random_forest"
    model_params: Dict[str, Any] = field(default_factory=lambda: {"n_estimators": 200})

    cv_folds: int = 5
    final_test: bool = False  # whether to run final test evaluation

    # ───────────────────────────────────────────────
    # TRAINING MODES
    # ───────────────────────────────────────────────
    use_optuna: bool = False  # auto-switch between simple and optuna
    simple_training: bool = True  # mode = simple by default

    # Optuna options:
    n_trials: int = 30
    optuna_timeout: int = 600
    optuna_direction: str = "maximize"
    optuna_sampler: str = "tpe"  # tpe | random | cmaes
    optuna_pruning: bool = False  # enable pruner


class DataConfig(BaseSettings):
    # ------------------------------------------------------------------
    # SAFE DEFAULT PATHS (project-relative, NOT absolute)
    # ------------------------------------------------------------------
    data_dir: Path = Field(
        default=Path("data/external"),
        description="Directory for raw data files (project-root relative).",
    )
    cache_dir: Path = Field(
        default=Path("data/processed"),
        description="Directory for processed & cached data (project-root relative).",
    )

    # ------------------------------------------------------------------
    # Dataset + processing options
    # ------------------------------------------------------------------
    dataset: str = Field(default="alexanderliapatis/miniboone", description="Kaggle dataset ID")

    test_size: float = Field(default=0.2, ge=0.1, le=0.3)
    val_size: float = Field(default=0.2, ge=0.1, le=0.3)

    random_state: int = Field(default=42, ge=0)
    target_col: str = Field(default="signal")
    scale_method: str = Field(default="standard")  # standard | robust | power

    variance_threshold: float = Field(default=1e-6, ge=0.0)
    add_outlier_flag: bool = Field(default=True)

    number_of_signals: int = Field(default=36499)
    number_of_background: int = Field(default=93565)

    use_cache: bool = Field(default=True)

    # ------------------------------------------------------------------
    # VALIDATION: ensure .data_dir is non-empty
    # ------------------------------------------------------------------
    @field_validator("data_dir", mode="before")
    @classmethod
    def validate_data_dir(cls, v):
        if v is None:
            raise ValueError("data_dir cannot be None")
        if isinstance(v, str) and v.strip() == "":
            raise ValueError("data_dir must be a non-empty string")
        return Path(v)

    # ------------------------------------------------------------------
    # RESOLVE PATHS RELATIVE TO PROJECT ROOT
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def resolve_paths(self) -> "DataConfig":
        """
        Ensures data_dir + cache_dir are resolved relative to project root if not absolute.
        Prevents accidental resolution to /data.
        """
        project_root = Path(__file__).resolve().parent.parent.parent  # repo root

        # Resolve data_dir
        if not self.data_dir.is_absolute():
            self.data_dir = (project_root / self.data_dir).resolve()

        # Resolve cache_dir
        if not self.cache_dir.is_absolute():
            self.cache_dir = (project_root / self.cache_dir).resolve()

        # Create directories if missing
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        return self

    # ------------------------------------------------------------------
    # VALIDATE SPLIT SIZES
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_split_sizes(self) -> "DataConfig":
        total_held_out = self.test_size + self.val_size
        if total_held_out > 0.5:
            raise ValueError(
                f"Combined test_size + val_size cannot exceed 0.5 " f"(got {total_held_out:.2f})."
            )
        return self

    model_config = {
        "env_file": ".env",
        "extra": "ignore",
        "validate_assignment": True,
    }
