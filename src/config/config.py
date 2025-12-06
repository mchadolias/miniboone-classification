from dataclasses import dataclass, field
from pathlib import Path
from typing import List

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


class DataConfig(BaseSettings):
    data_dir: Path = Field(
        default=Path("../data/external/"), description="Directory for data files"
    )
    dataset: str = Field(default="alexanderliapatis/miniboone", description="Kaggle dataset ID")
    test_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.3,
        description="Test set size ratio of the overall dataset (10-30%)",
    )
    val_size: float = Field(
        default=0.2,
        ge=0.1,
        le=0.3,
        description="Validation set size ratio of the overall dataset (10-30%).",
    )
    random_state: int = Field(default=42, ge=0, description="Random seed for reproducibility")
    target_col: str = Field(
        default="signal", description="Name of the target column in the dataset"
    )
    scale_method: str = Field(
        default="standard",
        description="Feature scaling method: 'standard', 'robust', or 'power'",
    )
    variance_threshold: float = Field(
        default=1e-6, ge=0.0, description="Variance threshold for feature selection"
    )
    add_outlier_flag: bool = Field(
        default=True, description="Whether to add an outlier flag feature"
    )
    number_of_signals: int = Field(
        default=36499, ge=0, description="Number of signal events in dataset"
    )
    number_of_background: int = Field(
        default=93565, ge=0, description="Number of background events in dataset"
    )
    use_cache: bool = Field(
        default=True, description="Whether to use cached processed data if available"
    )
    cache_dir: Path = Field(
        default=Path("../data/processed/"), description="Directory to store cached processed data"
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def validate_data_dir(cls, v):
        """Ensure data_dir is a non-empty string path before conversion to Path."""
        if v is None:
            raise ValueError("data_dir cannot be None")

        # Accept Path or str, but require non-empty
        if isinstance(v, Path):
            if str(v).strip() == "":
                raise ValueError("data_dir cannot be an empty path")
            return v

        if isinstance(v, str):
            if v.strip() == "":
                raise ValueError("data_dir must be a non-empty string")
            return Path(v)

        raise TypeError("data_dir must be a str or Path")

    @model_validator(mode="after")
    def validate_split_sizes(self) -> "DataConfig":
        """Validate that split sizes are reasonable and leave sufficient training data."""
        total_held_out = self.test_size + self.val_size
        if total_held_out > 0.5:
            raise ValueError(
                f"Combined test_size + val_size cannot exceed 0.5. "
                f"Got test_size={self.test_size}, val_size={self.val_size} (sum={total_held_out:.2f})"
            )

        train_size = 1 - self.test_size - self.val_size
        if train_size < 0.5:
            raise ValueError(
                f"Training set too small: {train_size:.1%}. " f"Need at least 50% training data."
            )

        return self

    model_config = {"env_file": ".env", "extra": "ignore", "validate_assignment": True}
