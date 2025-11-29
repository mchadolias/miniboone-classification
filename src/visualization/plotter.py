"""
Advanced plotting utilities for neutrino classification analysis.

This module provides publication-quality scientific visualization for
particle physics data analysis, with a focus on neutrino signal vs
background classification tasks.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# Import from your config
from src.config import BOXPLOT_PRESETS, VIOLIN_PRESETS, BoxplotConfig, SaveConfig, ViolinPlotConfig

# Handle optional dependencies
try:
    import scienceplots

    SCIENCEPLOTS_AVAILABLE = True
except ImportError:
    SCIENCEPLOTS_AVAILABLE = False
    warnings.warn("scienceplots not available, using default matplotlib styles")


def check_latex_available():
    """Check if LaTeX is actually available and working for matplotlib rendering."""
    original_usetex = plt.rcParams.get("text.usetex", False)

    try:
        # Temporarily enable LaTeX for testing
        plt.rcParams["text.usetex"] = True

        # Test with a simple figure
        fig, ax = plt.subplots(figsize=(1, 1), dpi=80)
        ax.text(0.5, 0.5, r"$\alpha$", transform=ax.transAxes, ha="center", va="center")

        # Force rendering without displaying
        fig.canvas.draw_idle()
        plt.close(fig)

        # If we get here, LaTeX works
        return True

    except Exception as e:
        print(f"[DEBUG] LaTeX test failed: {e}")
        return False
    finally:
        # Restore original setting
        plt.rcParams["text.usetex"] = original_usetex
        plt.close("all")  # Clean up any remaining figures


def setup_scientific_plotting(style: str = "science") -> None:
    """
    Setup scientific plotting style for publication-quality figures.

    Args:
        style: Plotting style ('science', 'seaborn', or 'default')
    """
    # Store original state
    original_usetex = plt.rcParams.get("text.usetex", False)

    try:
        if style == "science" and SCIENCEPLOTS_AVAILABLE:
            # Test LaTeX availability first
            latex_available = check_latex_available()

            if latex_available:
                plt.style.use(["science", "ieee", "grid"])
                plt.rcParams["text.usetex"] = True
                print("[STATUS] Using scienceplots style with LaTeX rendering")
            else:
                plt.style.use(["science", "ieee", "grid", "no-latex"])
                plt.rcParams["text.usetex"] = False
                print("[STATUS] Using scienceplots style without LaTeX rendering")

        elif style == "seaborn":
            plt.rcParams["text.usetex"] = False
            plt.style.use("seaborn-v0_8-whitegrid")
            print("[STATUS] Using seaborn style")
        else:
            plt.rcParams["text.usetex"] = False
            plt.style.use("default")
            print("[STATUS] Using default matplotlib style")

    except Exception as e:
        print(f"[WARN] Could not set style {style}: {e}")
        plt.style.use("default")
        plt.rcParams["text.usetex"] = original_usetex

    # Universal scientific styling
    plt.rcParams.update(
        {
            "figure.figsize": (10, 8),
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.family": "serif",
        }
    )

    # Configure math text if not using LaTeX
    if not plt.rcParams.get("text.usetex", False):
        plt.rcParams["mathtext.fontset"] = "stix"
        print("[STATUS] Using STIX fonts for math rendering")


class NeutrinoPlotter:
    """
    Advanced plotting utilities for neutrino classification analysis.

    This class provides comprehensive visualization capabilities for:
    - Feature distribution analysis (violin plots, boxplots, histograms)
    - Signal vs background comparisons
    - Correlation analysis and feature importance
    - Model performance evaluation
    - Publication-quality figure generation

    Attributes:
        style (str): Current plotting style
        corr (pd.DataFrame): Cached correlation matrix
    """

    def __init__(self, style: str = "science"):
        """
        Initialize the neutrino plotter.

        Args:
            style: Plotting style ('science', 'seaborn', or 'default')
        """
        self.style = style
        self.corr: Optional[pd.DataFrame] = None
        self._setup_plotting()

    def _setup_plotting(self) -> None:
        """Setup plotting style with appropriate fallbacks."""
        setup_scientific_plotting(self.style)

    def _validate_dataframe(self, df: pd.DataFrame, required_columns: List[str] = None) -> None:
        """
        Validate input dataframe for plotting.

        Args:
            df: DataFrame to validate
            required_columns: List of required column names

        Raises:
            TypeError: If input is not a DataFrame
            ValueError: If DataFrame is empty or missing required columns
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if df.empty:
            raise ValueError("DataFrame is empty")

        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

    def save_plot(self, fig: plt.Figure, save_config: SaveConfig, filename: str) -> None:
        """
        Save plot with specified format and settings.

        Args:
            fig: matplotlib Figure object to save
            save_config: SaveConfig with saving parameters
            filename: base filename (without extension)

        Raises:
            ValueError: If save_config is None
        """
        if save_config is None:
            raise ValueError("save_config cannot be None")

        try:
            os.makedirs(save_config.save_dir, exist_ok=True)
            full_path = os.path.join(save_config.save_dir, filename)

            successful_saves = 0
            for fmt in save_config.formats:
                save_path = f"{full_path}.{fmt}"
                try:
                    fig.savefig(
                        save_path,
                        dpi=save_config.dpi,
                        bbox_inches=save_config.bbox_inches,
                        facecolor="white",
                        transparent=save_config.transparent,
                    )
                    successful_saves += 1
                    print(f"[SUCCESS] Saved: {save_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to save {save_path}: {e}")

            if successful_saves == 0:
                print("[WARN]  No formats were successfully saved")

        except Exception as e:
            print(f"[ERROR] Error in save_plot: {e}")

    @staticmethod
    def calculate_smart_limits(
        df: pd.DataFrame, features: List[str], percentile: float = 1.0
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate limits excluding outliers using percentiles.
        Args:
            df: Input DataFrame
            features: List of feature names to calculate limits for
            percentile: Percentile to use for trimming (e.g., 1.0 = 1st and 99th percentiles)
        Returns:
            Dictionary mapping feature names to (lower, upper) limits
        """
        limits = {}
        for feature in features:
            data = df[feature].dropna()
            if len(data) == 0:
                limits[feature] = (0, 1)
                continue
            lower = np.percentile(data, percentile)
            upper = np.percentile(data, 100 - percentile)
            limits[feature] = (lower, upper)
        return limits

    # Update: Commented out due to bug and not serving much purpose
    # TODO: Debug it and make it useful - Priority: Low
    # def create_violin_boxplot_combo(
    #     self,
    #     df: pd.DataFrame,
    #     figsize: Tuple[int, int] = (20, 15),
    #     skip_signal: bool = True,
    #     config: Optional[ViolinPlotConfig] = None,
    #     preset: Optional[str] = None,
    #     save_config: Optional[SaveConfig] = None,
    #     **kwargs,
    # ) -> plt.Figure:
    #     """
    #     Create violin plots with embedded boxplots for distribution visualization.

    #     Args:
    #         df: Input DataFrame with features
    #         figsize: Figure size (width, height)
    #         skip_signal: Whether to skip the 'signal' column
    #         config: ViolinPlotConfig for styling
    #         preset: Preset configuration name
    #         save_config: Configuration for saving the plot
    #         **kwargs: Additional styling parameters

    #     Returns:
    #         matplotlib.Figure: The created figure

    #     Raises:
    #         ValueError: If preset is unknown
    #     """
    #     self._validate_dataframe(df)

    #     # Handle preset configurations
    #     if preset is not None:
    #         if preset not in VIOLIN_PRESETS:
    #             raise ValueError(
    #                 f"Unknown preset: {preset}. Available: {list(VIOLIN_PRESETS.keys())}"
    #             )
    #         config = VIOLIN_PRESETS[preset]

    #     # Use default config if none provided
    #     if config is None:
    #         config = ViolinPlotConfig()

    #     # Update config with any provided kwargs
    #     self._update_config_from_kwargs(config, kwargs)

    #     # Filter columns
    #     if skip_signal and "signal" in df.columns:
    #         plot_columns = [col for col in df.columns if col != "signal"]
    #     else:
    #         plot_columns = df.columns.tolist()

    #     n_cols = min(3, len(plot_columns))  # Max 3 columns
    #     n_rows = (len(plot_columns) + n_cols - 1) // n_cols

    #     fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    #     axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    #     for i, column in enumerate(plot_columns):
    #         if i >= len(axes):
    #             break

    #         data = df[column].dropna()
    #         if len(data) == 0:
    #             axes[i].text(
    #                 0.5, 0.5, "No Data", transform=axes[i].transAxes, ha="center", va="center"
    #             )
    #             axes[i].set_title(f"{column}", fontweight="bold")
    #             continue

    #         # Create violin plot
    #         try:
    #             sns.violinplot(
    #                 y=data,
    #                 ax=axes[i],
    #                 color=config.palette if isinstance(config.palette, str) else None,
    #                 inner=config.inner,
    #                 scale=config.scale,
    #                 bw=config.bw,
    #             )
    #         except Exception as e:
    #             print(f"Warning: Could not create violin plot for {column}: {e}")
    #             # Fallback to boxplot
    #             sns.boxplot(y=data, ax=axes[i])

    #         # Add summary statistics
    #         stats_text = self._format_stats_text(data)

    #         axes[i].set_title(
    #             f"{column}", fontweight="bold", fontsize=config.title_fontsize, pad=15
    #         )
    #         axes[i].text(
    #             0.02,
    #             0.98,
    #             stats_text,
    #             transform=axes[i].transAxes,
    #             fontsize=config.stats_fontsize,
    #             verticalalignment="top",
    #             bbox=dict(
    #                 boxstyle="round",
    #                 facecolor=config.bbox_facecolor,
    #                 alpha=config.bbox_alpha,
    #             ),
    #         )

    #         # Improve y-axis labels and spacing
    #         axes[i].tick_params(axis="y", labelsize=config.tick_fontsize)
    #         axes[i].set_ylabel("")

    #     # Hide empty subplots
    #     for i in range(len(plot_columns), len(axes)):
    #         axes[i].set_visible(False)

    #     plt.suptitle(
    #         "Feature Distribution Analysis: Violin Plots with Boxplots",
    #         fontsize=16,
    #         fontweight="bold",
    #         y=0.98,
    #     )
    #     plt.tight_layout(pad=3.0)

    #     # Save plot if save_config provided
    #     if save_config is not None:
    #         self.save_plot(fig, save_config, "violin_boxplot_combo")

    #     return fig

    def create_horizontal_boxplot_with_density(
        self,
        df: pd.DataFrame,
        config: Optional[BoxplotConfig] = None,
        preset: Optional[str] = None,
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create horizontal boxplots with distribution density plots.

        Args:
            df: Input DataFrame with features
            figsize: Figure size (width, height)
            config: BoxplotConfig for styling
            preset: Preset configuration name
            save_config: Configuration for saving the plot
            **kwargs: Additional parameters including:
                - max_features: Maximum number of features to plot (default: 30)
                - log_threshold: Orders of magnitude threshold for log scaling (default: 3.0)
                - auto_scale: Whether to automatically detect and apply scaling (default: True)
                - show_scale_indicator: Whether to show scale type indicator (default: True)
                - outlier_handling: How to handle outliers ('remove', 'clip', 'none') (default: 'clip')
                - outlier_percentile: Percentile for outlier detection (default: 1.0)
                - show_outlier_info: Whether to show outlier information (default: True)

        Returns:
            matplotlib.Figure: The created figure
        """
        self._validate_dataframe(df)

        # Handle preset configurations
        if preset is not None:
            if preset not in BOXPLOT_PRESETS:
                raise ValueError(
                    f"Unknown preset: {preset}. Available: {list(BOXPLOT_PRESETS.keys())}"
                )
            config = BOXPLOT_PRESETS[preset]

        # Use default config if none provided
        if config is None:
            config = BoxplotConfig()
        # Update config with any provided kwargs
        self._update_config_from_kwargs(config, kwargs)

        columns_to_plot = [col for col in df.columns if col != "signal"]
        max_features = kwargs.get("max_features", 30)
        auto_scale = kwargs.get("auto_scale", True)
        show_scale_indicator = kwargs.get("show_scale_indicator", True)
        outlier_handling = kwargs.get("outlier_handling", "clip")
        outlier_percentile = kwargs.get("outlier_percentile", 1.0)
        show_outlier_info = kwargs.get("show_outlier_info", True)

        # Limit number of features for performance
        if len(columns_to_plot) > max_features:
            print(f"[WARN] Limiting to first {max_features} features for performance")
            columns_to_plot = columns_to_plot[:max_features]

        fig, axes = plt.subplots(len(columns_to_plot), 1)

        if len(columns_to_plot) == 1:
            axes = [axes]

        for i, column in enumerate(columns_to_plot):
            original_data = df[column].dropna()
            if len(original_data) == 0:
                axes[i].text(
                    0.5, 0.5, "No Data", transform=axes[i].transAxes, ha="center", va="center"
                )
                axes[i].set_title(f"{column}", fontweight="bold")
                continue

            # Handle outliers
            data, outlier_info = self._handle_outliers(
                original_data, method=outlier_handling, percentile=outlier_percentile
            )

            # Create horizontal boxplot
            color = sns.color_palette(config.palette)[i % len(sns.color_palette(config.palette))]

            sns.boxplot(
                x=data,
                ax=axes[i],
                color=color,
                width=config.width,
                fliersize=config.fliersize,
                linewidth=config.linewidth,
            )

            # Create density plot
            ax2 = axes[i].twinx()
            try:
                sns.kdeplot(
                    data=data,
                    ax=ax2,
                    color=config.density_color,
                    alpha=config.density_alpha,
                    linewidth=config.density_linewidth,
                )
            except Exception as e:
                print(f"[WARN] Could not create density plot for {column}: {e}")

            # Auto-detect and apply appropriate scaling
            if auto_scale and len(data) > 0:
                scale_type = self._auto_detect_scale(
                    data, threshold=kwargs.get("log_threshold", 3.0)
                )

                if scale_type == "log":
                    axes[i].set_xscale("log")
                    ax2.set_xscale("log")
                    if show_scale_indicator:
                        axes[i].text(
                            0.02,
                            0.98,
                            "Log Scale",
                            transform=axes[i].transAxes,
                            fontsize=8,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
                        )

                elif scale_type == "symlog":
                    linthresh = self._calculate_linthresh(data)
                    axes[i].set_xscale(
                        "symlog", linthresh=linthresh, subs=[2, 3, 4, 5, 6, 7, 8, 9]
                    )
                    ax2.set_xscale("symlog", linthresh=linthresh, subs=[2, 3, 4, 5, 6, 7, 8, 9])
                    if show_scale_indicator:
                        axes[i].text(
                            0.02,
                            0.98,
                            "SymLog Scale",
                            transform=axes[i].transAxes,
                            fontsize=8,
                            verticalalignment="top",
                            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
                        )

            ax2.set_ylabel("Density", fontsize=9)
            ax2.grid(False)
            ax2.tick_params(axis="y", labelsize=8)

            # Style the main plot
            # axes[i].set_title(
            #     f"{column}",
            #     fontweight="bold",
            #     fontsize=config.title_fontsize,
            #     pad=10,
            # )
            axes[i].set_ylabel("")
            axes[i].tick_params(axis="x", labelsize=9)

            # Build statistics text with outlier information
            stats_lines = list(self._format_stats_text(data))

            if show_outlier_info and outlier_info["outliers_removed"] > 0:
                stats_lines.append(f"Outliers: {outlier_info['outliers_removed']}")
                if outlier_handling == "clip":
                    stats_lines.append(
                        f"Clipped at: [{outlier_info['lower_bound']:.2f}, {outlier_info['upper_bound']:.2f}]"
                    )

            stats_text = " | ".join(stats_lines)

            # axes[i].text(
            #     0.5,
            #     0.9,
            #     stats_text,
            #     transform=axes[i].transAxes,
            #     fontsize=config.stats_fontsize - 1,
            #     verticalalignment="top",
            #     ha="center",
            #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            # )

            # # Add data range information
            # range_text = f"Range: [{data.min():.2f}, {data.max():.2f}]"
            # if outlier_info["outliers_removed"] > 0:
            #     original_range = (
            #         f" (Original: [{original_data.min():.2f}, {original_data.max():.2f}])"
            #     )
            #     range_text += original_range

            # axes[i].text(
            #     0.5,
            #     0.75,
            #     range_text,
            #     transform=axes[i].transAxes,
            #     fontsize=8,
            #     verticalalignment="top",
            #     ha="center",
            # )
        plt.tight_layout()

        # Save plot if save_config provided
        if save_config is not None:
            self.save_plot(fig, save_config, "horizontal_boxplot_density")

        return fig

    def _handle_outliers(
        self, data: pd.Series, method: str = "clip", percentile: float = 1.0
    ) -> Tuple[pd.Series, Dict]:
        """
        Handle outliers in data using specified method.
        """
        if method == "none" or len(data) == 0:
            return data, {
                "outliers_removed": 0,
                "lower_bound": data.min() if len(data) > 0 else 0,
                "upper_bound": data.max() if len(data) > 0 else 0,
                "method": "none",
            }

        # Calculate bounds using percentiles
        lower_bound = np.percentile(data, percentile)
        upper_bound = np.percentile(data, 100 - percentile)

        outlier_info = {
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "method": method,
            "outliers_removed": 0,
        }

        if method == "remove":
            filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
            outlier_info["outliers_removed"] = len(data) - len(filtered_data)
            return filtered_data, outlier_info

        elif method == "clip":
            clipped_data = data.clip(lower=lower_bound, upper=upper_bound)
            outliers_mask = (data < lower_bound) | (data > upper_bound)
            outlier_info["outliers_removed"] = outliers_mask.sum()
            return clipped_data, outlier_info

        else:
            print(f"[WARN] Unknown outlier handling method: {method}. Using 'none'.")
            return data, outlier_info

    def _auto_detect_scale(self, data: pd.Series, threshold: float = 3.0) -> str:
        """
        Automatically detect whether to use linear, log, or symlog scale.

        Args:
            data: Input data series
            threshold: Orders of magnitude threshold for log scaling

        Returns:
            Scale type: 'linear', 'log', or 'symlog'
        """
        if len(data) == 0:
            return "linear"

        # Remove zeros, infinities, and NaN for analysis
        clean_data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_data) == 0:
            return "linear"

        has_negatives = (clean_data < 0).any()
        has_positives = (clean_data > 0).any()
        has_zeros = (clean_data == 0).any()

        # Case 1: Only positive values (or positive + zeros)
        if has_positives and not has_negatives:
            positive_data = clean_data[clean_data > 0]
            if len(positive_data) > 1:
                # Avoid log(0) by using min positive value
                data_min = positive_data.min()
                data_max = positive_data.max()
                if data_min > 0 and data_max > 0:
                    data_range = np.log10(data_max) - np.log10(data_min)
                    if data_range > threshold:
                        return "log"

        # Case 2: Only negative values (or negative + zeros)
        elif has_negatives and not has_positives:
            negative_data = clean_data[clean_data < 0]
            if len(negative_data) > 1:
                # Use absolute values for negative data
                abs_min = (-negative_data).min()
                abs_max = (-negative_data).max()
                if abs_min > 0 and abs_max > 0:
                    data_range = np.log10(abs_max) - np.log10(abs_min)
                    if data_range > threshold:
                        return "log"

        # Case 3: Mixed positive and negative values
        elif has_negatives and has_positives:
            # Check if data spans a wide dynamic range in both directions
            # Use absolute values of non-zero data
            nonzero_data = clean_data[clean_data != 0]
            if len(nonzero_data) > 1:
                abs_values = np.abs(nonzero_data)
                data_range = np.log10(abs_values.max()) - np.log10(abs_values.min())
                if data_range > threshold:
                    return "symlog"

        return "linear"

    def _calculate_linthresh(self, data: pd.Series, percentile: float = 25) -> float:
        """
        Calculate appropriate linear threshold for symlog scale.

        The linear threshold defines the range around zero where the scale remains linear.

        Args:
            data: Input data series
            percentile: Percentile to use for calculating threshold (default: 25th percentile)

        Returns:
            Linear threshold value
        """
        if len(data) == 0:
            return 1e-3

        # Use absolute values of non-zero data
        nonzero_data = data[data != 0]
        if len(nonzero_data) == 0:
            return 1e-3

        abs_data = np.abs(nonzero_data)

        # Calculate threshold based on percentile of absolute values
        # This ensures the linear region covers typical small values
        threshold = np.percentile(abs_data, percentile) * 0.1

        # Ensure threshold is reasonable (not too small or too large)
        threshold = max(1e-6, min(threshold, 1.0))

        return threshold

    def create_signal_vs_background_distributions(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        feature_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        n_cols: int = 5,
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Compare distributions of features between signal and background.

        Args:
            df: Input dataframe with features and signal column
            features: List of features to plot. If None, uses all numeric features except 'signal'
            feature_limits: Dictionary mapping feature names to (xmin, xmax) limits
            n_cols: Number of columns in the subplot grid
            save_config: Configuration for saving the plot
            **kwargs: Additional parameters including colors, alpha, show_stats

        Returns:
            matplotlib.Figure: The created figure

        Raises:
            ValueError: If 'signal' column is missing
        """
        self._validate_dataframe(df, required_columns=["signal"])

        if features is None:
            features = [
                col for col in df.select_dtypes(include=[np.number]).columns if col != "signal"
            ]

        # Filter out features with no variance
        features = [f for f in features if df[f].nunique() > 1]

        if not features:
            raise ValueError("No valid features found for plotting")

        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        figsize = kwargs.get("figsize", (15, 3 * n_rows))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        # Customization parameters
        colors = kwargs.get("colors", {"signal": "blue", "background": "red"})
        alpha = kwargs.get("alpha", 0.7)
        show_stats = kwargs.get("show_stats", True)

        for i, feature in enumerate(features):
            if i >= len(axes):
                break

            # Plot distributions for signal vs background
            signal_data = df[df["signal"] == 1][feature].dropna()
            background_data = df[df["signal"] == 0][feature].dropna()

            if len(signal_data) == 0 or len(background_data) == 0:
                axes[i].text(
                    0.5,
                    0.5,
                    "Insufficient Data",
                    transform=axes[i].transAxes,
                    ha="center",
                    va="center",
                )
                axes[i].set_title(f"{feature}", fontweight="bold")
                continue

            # KDE plots
            sns.kdeplot(
                background_data,
                ax=axes[i],
                label=r"Background ($\nu_{\mu}$)",
                color=colors["background"],
                alpha=alpha,
            )

            sns.kdeplot(
                signal_data,
                ax=axes[i],
                label=r"Signal ($\nu_{e}$)",
                color=colors["signal"],
                alpha=alpha,
            )

            # Apply custom limits if specified for this feature
            if feature_limits and feature in feature_limits:
                xmin, xmax = feature_limits[feature]
                axes[i].set_xlim(xmin, xmax)

            # Add vertical padding for KDE plots
            y_min, y_max = axes[i].get_ylim()
            y_padding = (y_max - y_min) * 0.10  # 10% padding
            axes[i].set_ylim(y_min, y_max + y_padding)
            axes[i].set_title(f"{feature}", fontweight="bold", fontsize=11)
            axes[i].set_xlabel("")
            axes[i].legend(fontsize=9, frameon=False)

            # Add statistical comparison
            if show_stats and len(signal_data) > 0 and len(background_data) > 0:
                try:
                    t_stat, p_value = stats.ttest_ind(
                        signal_data, background_data, equal_var=False
                    )
                    axes[i].text(
                        0.03,
                        0.96,
                        f"p-value: {p_value:.2e}",
                        transform=axes[i].transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        # bbox=dict(facecolor="white", alpha=0.8),
                    )
                except Exception as e:
                    print(f"Warning: Could not compute statistics for {feature}: {e}")

        # Hide empty subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()

        # Save plot if save_config provided
        if save_config is not None:
            self.save_plot(fig, save_config, "signal_background_distributions")

        return fig

    def create_top_correlation_map(
        self,
        df: pd.DataFrame,
        num_features: int = 10,
        mask_upper: bool = True,
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot a correlation map for the top correlated features.

        Args:
            df: Input DataFrame
            num_features: Number of top features to include
            mask_upper: Whether to mask the upper triangle
            save_config: Configuration for saving the plot
            **kwargs: Additional parameters for seaborn.heatmap

        Returns:
            matplotlib.Figure: The created figure
        """
        self._validate_dataframe(df)

        # Compute correlation map
        corr = self._compute_correlation(df)

        if "signal" not in corr.columns:
            raise ValueError("'signal' column not found in correlation matrix")

        # Get top features correlated with signal
        target_corrs = corr["signal"].sort_values(ascending=False)
        top_features = target_corrs.head(num_features).index.tolist()

        if not top_features:
            raise ValueError("No features found for correlation plot")

        corr_top = corr.loc[top_features, top_features]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create mask only if requested
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr_top, dtype=bool))

        # Create heatmap
        sns.heatmap(
            corr_top,
            annot=True,
            fmt=".2f",
            ax=ax,
            cmap="RdBu_r",
            center=0,
            annot_kws={"size": 10},
            square=True,
            mask=mask,
            cbar_kws={"shrink": 0.8},
            **kwargs,
        )

        plt.title(f"Top {num_features} Features Correlated with Signal", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save plot if save_config provided
        if save_config is not None:
            self.save_plot(fig, save_config, f"top_{num_features}_correlation_map")

        return fig

    def create_full_correlation_map(
        self,
        df: pd.DataFrame,
        mask_upper: bool = True,
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot full correlation map for all features.

        Args:
            df: Input DataFrame
            mask_upper: Whether to mask the upper triangle
            save_config: Configuration for saving the plot
            **kwargs: Additional parameters for seaborn.heatmap

        Returns:
            matplotlib.Figure: The created figure
        """
        self._validate_dataframe(df)

        # Compute correlation map
        corr = self._compute_correlation(df)

        # Create figure
        figsize = kwargs.get("figsize", (14, 12))
        fig, ax = plt.subplots(figsize=figsize)

        # Create mask only if requested
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool))

        # Optimize the large heatmap
        sns.heatmap(
            corr,
            annot=False,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            cbar_kws={"shrink": 0.8},
            square=True,
            mask=mask,
            **kwargs,
        )

        # Add feature names with rotation
        plt.xticks(rotation=90, ha="center", fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.title("Full Correlation Matrix", fontsize=16, pad=20)
        plt.tight_layout()

        # Save plot if save_config provided
        if save_config is not None:
            self.save_plot(fig, save_config, "full_correlation_map")

        return fig

    def _compute_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Pearson correlation for numeric features.

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: Correlation matrix
        """
        # Select only numeric columns to avoid errors
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return pd.DataFrame()

        self.corr = numeric_df.corr()
        return self.corr

    def create_top_feature_pairplot(
        self,
        df: pd.DataFrame,
        num_features: int = 8,
        figsize: Tuple[int, int] = (12, 10),
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create pairplot of top correlated features.

        Args:
            df: Input DataFrame
            num_features: Number of top features to include
            figsize: Figure size (width, height)
            save_config: Configuration for saving the plot
            **kwargs: Additional parameters for seaborn.pairplot

        Returns:
            matplotlib.Figure: The created figure
        """
        self._validate_dataframe(df)

        # Compute correlations
        corr = self._compute_correlation(df)

        if "signal" not in corr.columns:
            raise ValueError("'signal' column not found for pairplot")

        # Get top features correlated with target
        top_features = (
            corr["signal"].abs().sort_values(ascending=False).head(num_features).index.tolist()
        )

        if "signal" not in top_features and "signal" in df.columns:
            top_features.append("signal")

        # Pairplot of only important features
        try:
            pairplot = sns.pairplot(
                df[top_features],
                hue="signal" if "signal" in df.columns else None,
                palette=["red", "blue"],
                plot_kws={"alpha": 0.6, "s": 10},
                diag_kws={"alpha": 0.7},
                corner=True,
                **kwargs,
            )

            # Adjust the figure size
            pairplot.fig.set_size_inches(figsize)
            plt.suptitle(f"Pairplot of Top {num_features} Features", y=1.02, fontsize=16)

            # Save plot if save_config provided
            if save_config is not None:
                self.save_plot(pairplot.fig, save_config, f"top_{num_features}_feature_pairplot")

            return pairplot.fig

        except Exception as e:
            print(f"Error creating pairplot: {e}")
            # Create a simple fallback plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "Could not create pairplot",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
            return fig

    def create_correlation_barplot(
        self,
        df: pd.DataFrame,
        num_features: int = 20,
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create barplot showing feature correlations with target.

        Args:
            df: Input DataFrame
            num_features: Number of features to show
            save_config: Configuration for saving the plot
            **kwargs: Additional parameters

        Returns:
            matplotlib.Figure: The created figure
        """
        self._validate_dataframe(df)

        fig, ax = plt.subplots(figsize=(10, 8))
        corr = self._compute_correlation(df)

        if "signal" not in corr.columns:
            raise ValueError("'signal' column not found for correlation barplot")

        corr_with_target = (
            corr["signal"].drop("signal", errors="ignore").sort_values(ascending=False)
        )

        # Take top and bottom features for balanced view
        top_pos = corr_with_target.head(num_features // 2)
        top_neg = corr_with_target.tail(num_features // 2)
        combined = pd.concat([top_pos, top_neg])

        # Create horizontal bar plot
        y_pos = np.arange(len(combined))
        colors = ["blue" if x >= 0 else "red" for x in combined.values]

        ax.barh(y_pos, combined.values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(combined.index)
        ax.set_xlabel("Correlation Coefficient")
        ax.set_title(f"Top {num_features} Feature Correlations with Signal")
        ax.grid(axis="x", alpha=0.3)

        # Add value annotations
        for i, v in enumerate(combined.values):
            ax.text(v, i, f"{v:.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)

        fig.tight_layout()

        # Save plot if save_config provided
        if save_config is not None:
            self.save_plot(fig, save_config, f"correlation_barplot_{num_features}_features")

        return fig

    def create_target_distribution_plot(
        self,
        df: pd.DataFrame,
        column: str = "signal",
        stat: str = "count",  # Fixed parameter name from 'stats' to 'stat'
        show_legend: bool = False,
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create distribution plot for target variable (binary: 1/0 or True/False).

        Args:
            df: Input DataFrame
            column: Target column name (binary: 1/0, True/False, or signal/background)
            stat: Which statistic to use for comparison ("count", "percent")  # Fixed docstring
            show_legend: Whether to show the legend
            save_config: Configuration for saving the plot
            **kwargs: Additional parameters for seaborn plotting

        Returns:
            matplotlib.Figure: The created figure
        """
        self._validate_dataframe(df, required_columns=[column])

        # Create a copy to avoid modifying original data
        plot_df = df.copy()

        # Map values to consistent labels for plotting
        unique_values = plot_df[column].unique()

        if len(unique_values) != 2:
            raise ValueError(
                f"Column '{column}' must be binary (2 unique values), found {len(unique_values)}"
            )

        if set(unique_values) == {0, 1}:
            sorted_vals = sorted(unique_values)
            plot_df["plot_label"] = plot_df[column].map(
                {
                    sorted_vals[0]: r"Background ($\nu_{\mu}$)",
                    sorted_vals[1]: r"Signal ($\nu_{e}$)",
                }
            )
            hue_order = [r"Background ($\nu_{\mu}$)", r"Signal ($\nu_{e}$)"]

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (10, 8)))

        # Use seaborn countplot with hue instead of palette for x
        sns.countplot(
            data=plot_df,
            x="plot_label",
            hue="plot_label",  # Add hue for coloring
            ax=ax,
            palette=["black", "red"],
            order=hue_order,
            hue_order=hue_order,
            stat=stat,  # Use the correct parameter name
            legend=False,  # Disable seaborn's automatic legend
            **kwargs,
        )

        # Add annotations with counts and percentages
        total = len(plot_df)
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            if stat == "percent":
                # For percent stat, height is already percentage
                percentage = height
                count = (height / 100) * total
                annotation_text = f"{percentage:.2f}\\%"
            else:
                # For count stat, height is count
                count = height
                percentage = (count / total) * 100
                annotation_text = f"{count:,} ({percentage:.2f}\\%)"

            ax.text(
                p.get_x() + p.get_width() / 2.0,
                height + max(ax.get_ylim()) * 0.01,
                annotation_text,
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        # Customize plot
        ax.set_xlabel("")
        if stat == "count":
            ax.set_ylabel("Count")
        elif stat == "percent":
            ax.set_ylabel(r"Percent ($\%$)")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)  # Add space for annotations
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        # Add custom legend if requested
        if show_legend:
            legend_elements = [
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor="black", alpha=0.8, label=r"Background ($\nu_{\mu}$)"
                ),
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor="red", alpha=0.8, label=r"Signal ($\nu_{e}$)"
                ),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

        fig.tight_layout()

        if save_config is not None:
            self.save_plot(fig, save_config, f"target_distribution_{column}")

        return fig

    def _update_config_from_kwargs(self, config: Any, kwargs: Dict[str, Any]) -> None:
        """
        Update config object with kwargs.

        Args:
            config: Configuration object to update
            kwargs: Dictionary of key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def _format_stats_text(self, data: pd.Series) -> str:
        """
        Helper method to format statistics text.

        Args:
            data: Input data series

        Returns:
            str: Formatted statistics text
        """
        if len(data) == 0:
            return "No data"

        return f"""N: {len(data):,}
Mean: {data.mean():.2f}
Std: {data.std():.2f}
Min: {data.min():.2f}
Max: {data.max():.2f}
Skew: {data.skew():.2f}"""
