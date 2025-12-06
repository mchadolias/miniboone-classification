from typing import List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import SaveConfig
from src.styles.plot_style import setup_scientific_plotting
from src.utils.logger import get_global_logger

logger = get_global_logger(__name__)


class ScientificPlotter:
    """
    Base plotting utility for physics and scientific visualization workflows.

    This class sets up consistent figure styles, validation methods, and safe
    export utilities. It serves as the foundation for domain-specific plotters
    such as `NeutrinoPlotter` (signal-background visualization) or
    `DimensionalityReductionPlotter` (feature space embeddings).

    Attributes
    ----------
    style : str
        The matplotlib style preset to use (e.g., "science", "seaborn", "default").
    """

    def __init__(self, style: str = "science") -> None:
        """Initialize the ScientificPlotter with a chosen visualization style."""
        self.style = style
        setup_scientific_plotting(style)
        logger.info(f"ScientificPlotter initialized with style: '{style}'")

    # -------------------------------------------------------------------------
    # Validation utilities
    # -------------------------------------------------------------------------
    def validate_dataframe(
        self, df: pd.DataFrame, required_cols: Optional[List[str]] = None
    ) -> None:
        """Ensure the input DataFrame is valid and contains required columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.
        required_cols : Optional[List[str]]
            A list of column names that must exist in the DataFrame.

        Raises
        ------
        TypeError
            If df is not a pandas DataFrame.
        ValueError
            If df is empty or missing required columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        if required_cols:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

    # -------------------------------------------------------------------------
    # Figure export utilities
    # -------------------------------------------------------------------------
    def export_figure(self, fig: plt.Figure, save_config: SaveConfig, filename: str) -> None:
        """Save a matplotlib figure with consistent parameters.

        Parameters
        ----------
        fig : plt.Figure
            Matplotlib figure to save.
        save_config : SaveConfig
            Configuration for saving plots (output dir, formats, DPI, etc.).
        filename : str
            Base name for saved files (without extension).

        Notes
        -----
        This method supports multi-format saving (e.g., PNG + PDF)
        and ensures output directories exist.
        """
        if save_config is None:
            raise ValueError("save_config cannot be None.")

        save_dir = Path(save_config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for fmt in save_config.formats:
            path = save_dir / f"{filename}.{fmt}"
            try:
                fig.savefig(
                    path,
                    dpi=save_config.dpi,
                    bbox_inches=save_config.bbox_inches,
                    facecolor="white",
                    transparent=save_config.transparent,
                )
                logger.info(f"Figure saved: {path}")
            except Exception as e:
                logger.error(f"Failed to save figure '{path}': {e}")

    # -------------------------------------------------------------------------
    # Statistical annotations
    # -------------------------------------------------------------------------
    def annotate_statistics(self, ax: plt.Axes, data: pd.Series) -> None:
        """Annotate subplot with simple statistics (mean, std, count).

        Parameters
        ----------
        ax : plt.Axes
            The subplot axis to annotate.
        data : pd.Series
            The numeric data used for computing statistics.
        """
        if data.empty:
            logger.warning("Empty data passed to annotate_statistics(). Skipping.")
            return

        text = f"N = {len(data)}\nMean = {data.mean():.2f}\nStd = {data.std():.2f}"
        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # -------------------------------------------------------------------------
    # Optional base plotting helper
    # -------------------------------------------------------------------------
    def plot_distribution(
        self, data: pd.Series, ax: Optional[plt.Axes] = None, bins: int = 50, color: str = "gray"
    ) -> plt.Axes:
        """Quick helper for plotting a histogram with KDE.

        Parameters
        ----------
        data : pd.Series
            The data to plot.
        ax : Optional[plt.Axes]
            The axis on which to plot. If None, creates a new one.
        bins : int
            Number of histogram bins.
        color : str
            Color for the histogram and KDE curve.

        Returns
        -------
        plt.Axes
            The axis containing the plot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        sns.histplot(data, bins=bins, kde=True, color=color, alpha=0.6, ax=ax)
        self.annotate_statistics(ax, data)
        ax.set_xlabel(data.name or "Value")
        ax.set_ylabel("Density")
        return ax
