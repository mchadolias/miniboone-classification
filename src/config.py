# src/config.py
from dataclasses import dataclass, field
from typing import List

from pydantic_settings import BaseSettings


class DataConfig(BaseSettings):
    data_dir: str = "data"
    dataset: str = "alexanderliapatis/miniboone"
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    number_of_signals: int = 36499
    number_of_background: int = 93565

    model_config = {"env_file": ".env", "extra": "ignore"}


@dataclass
class SaveConfig:
    save_dir: str = "./figures"
    formats: List[str] = field(default_factory=lambda: ["png", "pdf"])
    dpi: int = 300
    transparent: bool = False
    bbox_inches: str = "tight"


@dataclass
class ViolinPlotConfig:
    palette: str = "pastel"
    inner: str = "box"
    scale: str = "width"
    bw: float = 0.2
    title_fontsize: int = 12
    stats_fontsize: int = 9
    tick_fontsize: int = 9
    bbox_facecolor: str = "lightyellow"
    bbox_alpha: float = 0.8


@dataclass
class BoxplotConfig:
    palette: str = "viridis"
    width: float = 0.7
    fliersize: float = 5.0
    linewidth: float = 1.5
    density_color: str = "darkred"
    density_alpha: float = 0.6
    density_linewidth: float = 2.0
    title_fontsize: int = 11
    stats_fontsize: int = 9


@dataclass
class CorrelationConfig:
    cmap: str = "RdBu_r"
    center: float = 0.0
    annot: bool = True
    annot_fontsize: int = 8
    square: bool = True


@dataclass
class DistributionConfig:
    signal_color: str = "blue"
    background_color: str = "red"
    signal_alpha: float = 0.7
    background_alpha: float = 0.7
    linewidth: float = 2.0


# Simple presets
VIOLIN_PRESETS = {
    "default": ViolinPlotConfig(),
    "publication": ViolinPlotConfig(
        title_fontsize=14, stats_fontsize=10, bbox_facecolor="white", bbox_alpha=0.9
    ),
}

BOXPLOT_PRESETS = {
    "default": BoxplotConfig(),
    "high_contrast": BoxplotConfig(palette="dark", density_color="black", density_alpha=0.8),
}

CORRELATION_PRESETS = {
    "default": CorrelationConfig(),
    "publication": CorrelationConfig(annot_fontsize=10, cmap="coolwarm"),
}
