# src/config.py
from pydantic_settings import BaseSettings
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


class DataConfig(BaseSettings):
    data_dir: str = "data"
    dataset: str = "alexanderliapatis/miniboone"
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    number_of_signals: int = 36499
    number_of_background: int = 93565  # contained in the original dataset from UC Irvine

    class Config:
        env_file = ".env"


@dataclass
class ViolinPlotConfig:
    """Configuration for violin plot parameters"""

    palette: str = "pastel"
    inner: str = "box"
    linewidth: float = 1
    saturation: float = 0.75
    bw_method: Optional[str] = None
    cut: int = 2
    title_fontsize: int = 12
    stats_fontsize: int = 8
    tick_fontsize: int = 9
    bbox_facecolor: str = "wheat"
    bbox_alpha: float = 0.8
    extra_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for seaborn plotting"""
        base_dict = asdict(self)
        non_plot_params = {
            "title_fontsize",
            "stats_fontsize",
            "tick_fontsize",
            "bbox_facecolor",
            "bbox_alpha",
            "extra_kwargs",
        }
        plot_dict = {k: v for k, v in base_dict.items() if k not in non_plot_params}
        plot_dict.update(self.extra_kwargs)
        return plot_dict


@dataclass
class BoxplotConfig:
    """Configuration for boxplot with density parameters"""

    palette: str = "Set2"
    width: float = 0.7
    fliersize: int = 3
    linewidth: float = 1.5
    density_color: str = "darkblue"
    density_alpha: float = 0.5
    density_linewidth: float = 1.5
    title_fontsize: int = 12
    stats_fontsize: int = 9
    density_fontsize: int = 9
    boxplot_kwargs: Optional[Dict[str, Any]] = None
    kdeplot_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.boxplot_kwargs is None:
            self.boxplot_kwargs = {}
        if self.kdeplot_kwargs is None:
            self.kdeplot_kwargs = {}


# You can also add preset configurations
VIOLIN_PRESETS = {
    "default": ViolinPlotConfig(),
    "publication": ViolinPlotConfig(
        palette="viridis",
        inner="quartile",
        linewidth=1.5,
        title_fontsize=14,
        stats_fontsize=10,
        bbox_facecolor="lightgray",
    ),
    "minimal": ViolinPlotConfig(
        inner=None, linewidth=0.5, stats_fontsize=0  # No inner plot  # Hide stats
    ),
}

BOXPLOT_PRESETS = {
    "default": BoxplotConfig(),
    "high_contrast": BoxplotConfig(
        palette="dark", density_color="red", density_alpha=0.7, linewidth=2
    ),
}
