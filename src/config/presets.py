from dataclasses import dataclass


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
