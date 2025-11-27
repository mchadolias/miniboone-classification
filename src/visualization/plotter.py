# src/plotter.py
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# Import from your config
from src.config import BOXPLOT_PRESETS, VIOLIN_PRESETS, BoxplotConfig, SaveConfig, ViolinPlotConfig


def calculate_smart_limits(df, features, percentile=1):
    """Calculate limits excluding outliers"""
    limits = {}
    for feature in features:
        data = df[feature].dropna()
        lower = np.percentile(data, percentile)
        upper = np.percentile(data, 100 - percentile)
        limits[feature] = (lower, upper)
    return limits


class NeutrinoPlotter:
    """
    Advanced plotting utilities for neutrino classification analysis
    """

    def __init__(self, style: str = "seaborn-v0_8"):
        self.style = style
        plt.style.use(style)
        self.corr: pd.DataFrame

    def save_plot(self, fig: plt.Figure, save_config: SaveConfig, filename: str) -> None:
        """
        Save plot with specified format and settings
        """
        if not os.path.exists(save_config.save_dir):
            os.makedirs(save_config.save_dir)

        full_path = os.path.join(save_config.save_dir, filename)

        for fmt in save_config.formats:
            save_path = f"{full_path}.{fmt}"
            fig.savefig(
                save_path,
                dpi=save_config.dpi,
                bbox_inches=save_config.bbox_inches,
                facecolor="white",
                transparent=save_config.transparent,
            )
            print(f"Saved: {save_path}")

    def create_violin_boxplot_combo(
        self,
        df: pd.DataFrame,
        figsize: tuple = (22, 16),
        skip_signal: bool = True,
        config: Optional[ViolinPlotConfig] = None,
        preset: Optional[str] = None,
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create violin plots with embedded boxplots for better distribution visualization
        """
        # Handle preset configurations
        if preset is not None:
            if preset not in VIOLIN_PRESETS:
                raise ValueError(
                    f"Unknown preset: {preset}. Available: {list(VIOLIN_PRESETS.keys())}"
                )
            config = VIOLIN_PRESETS[preset]

        # Use default config if none provided
        if config is None:
            config = ViolinPlotConfig()

        # Update config with any provided kwargs
        self._update_config_from_kwargs(config, kwargs)

        # Filter columns
        if skip_signal and "signal" in df.columns:
            plot_columns = [col for col in df.columns if col != "signal"]
        else:
            plot_columns = df.columns.tolist()

        n_cols = len(plot_columns)
        n_rows = (n_cols + 2) // 3

        fig, axes = plt.subplots(n_rows, 3, figsize=figsize)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, column in enumerate(plot_columns):
            if i >= len(axes):
                break

            data = df[column].dropna()

            # Create violin plot with improved spacing
            sns.violinplot(
                y=data,
                ax=axes[i],
                palette=config.palette,
                inner=config.inner,
                scale=config.scale,
                bw=config.bw,
            )

            # Add summary statistics with config-based styling
            stats_text = self._format_stats_text(data)

            axes[i].set_title(
                f"{column}", fontweight="bold", fontsize=config.title_fontsize, pad=15
            )
            axes[i].text(
                0.02,
                0.98,
                stats_text,
                transform=axes[i].transAxes,
                fontsize=config.stats_fontsize,
                verticalalignment="top",
                bbox=dict(
                    boxstyle="round",
                    facecolor=config.bbox_facecolor,
                    alpha=config.bbox_alpha,
                ),
            )

            # Improve y-axis labels and spacing
            axes[i].tick_params(axis="y", labelsize=config.tick_fontsize)
            axes[i].set_ylabel("")

        # Hide empty subplots
        for i in range(len(plot_columns), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(
            "Feature Distribution Analysis: Violin Plots with Boxplots",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout(pad=3.0)

        # Save plot if save_config provided
        if save_config is not None:
            self.save_plot(fig, save_config, "violin_boxplot_combo")

        return fig

    def create_horizontal_boxplot_with_density(
        self,
        df: pd.DataFrame,
        figsize: tuple = (12, 25),
        config: Optional[BoxplotConfig] = None,
        preset: Optional[str] = None,
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create horizontal boxplots with distribution density plots
        """
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

        # Limit number of features for performance
        if len(columns_to_plot) > max_features:
            print(f"Warning: Limiting to first {max_features} features for performance")
            columns_to_plot = columns_to_plot[:max_features]

        n_cols = len(columns_to_plot)

        fig, axes = plt.subplots(n_cols, 1, figsize=figsize)

        if n_cols == 1:
            axes = [axes]

        for i, column in enumerate(columns_to_plot):
            data = df[column].dropna()

            # Create horizontal boxplot with config - FIXED: no boxplot_kwargs
            color = sns.color_palette(config.palette)[i % len(sns.color_palette(config.palette))]

            # Use direct config attributes instead of boxplot_kwargs
            sns.boxplot(
                x=data,
                ax=axes[i],
                color=color,
                width=config.width,
                fliersize=config.fliersize,
                linewidth=config.linewidth,
            )

            # Create density plot with config - FIXED: no kdeplot_kwargs
            ax2 = axes[i].twinx()
            sns.kdeplot(
                data=data,
                ax=ax2,
                color=config.density_color,
                alpha=config.density_alpha,
                linewidth=config.density_linewidth,
            )
            ax2.set_ylabel("Density", fontsize=9)
            ax2.grid(False)
            ax2.tick_params(axis="y", labelsize=8)

            # Style the main plot
            axes[i].set_title(
                f"{column}",
                fontweight="bold",
                fontsize=config.title_fontsize,
                pad=10,
            )
            axes[i].set_ylabel("")
            axes[i].tick_params(axis="x", labelsize=9)

            # Add statistical information
            stats_text = (
                f"Mean: {data.mean():.2f} | Median: {data.median():.2f} | Std: {data.std():.2f}"
            )
            axes[i].text(
                0.7,
                0.9,
                stats_text,
                transform=axes[i].transAxes,
                fontsize=config.stats_fontsize,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

            # Add sample size and skewness
            axes[i].text(
                0.7,
                0.75,
                f"N = {len(data):,} | Skew: {data.skew():.2f}",
                transform=axes[i].transAxes,
                fontsize=8,
                verticalalignment="top",
            )

        plt.suptitle(
            "Feature Distributions: Boxplots with Density Overlay",
            fontsize=16,
            fontweight="bold",
            y=0.95,
        )
        plt.tight_layout()

        # Save plot if save_config provided
        if save_config is not None:
            self.save_plot(fig, save_config, "horizontal_boxplot_density")

        return fig

    def create_signal_vs_background_distributions(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        feature_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        n_cols: int = 5,
        figsize: tuple = (20, 15),
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Compare distributions of features between signal and background

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with features and signal column
        features : List[str], optional
            List of features to plot. If None, uses all numeric features except 'signal'
        feature_limits : Dict[str, Tuple[float, float]], optional
            Dictionary mapping feature names to (xmin, xmax) limits for each subplot
            Example: {'feature1': (0, 10), 'feature2': (-5, 5)}
        n_cols : int
            Number of columns in the subplot grid
        figsize : tuple
            Figure size
        save_config : SaveConfig, optional
            Configuration for saving the plot
        **kwargs :
            Additional arguments
        """
        if "signal" not in df.columns:
            raise ValueError("DataFrame must contain 'signal' column")

        if features is None:
            features = [
                col for col in df.select_dtypes(include=[np.number]).columns if col != "signal"
            ]

        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, feature in enumerate(features):
            if i >= len(axes):
                break

            # Plot distributions for signal vs background
            signal_data = df[df["signal"] == 1][feature].dropna()
            background_data = df[df["signal"] == 0][feature].dropna()

            # KDE plots
            sns.kdeplot(
                background_data,
                ax=axes[i],
                label=r"Background ($\nu_{\mu}$)",
                color="red",
                alpha=0.7,
            )

            sns.kdeplot(
                signal_data, ax=axes[i], label=r"Signal ($\nu_{e}$)", color="blue", alpha=0.7
            )

            # Apply custom limits if specified for this feature
            if feature_limits and feature in feature_limits:
                xmin, xmax = feature_limits[feature]
                axes[i].set_xlim(xmin, xmax)

            axes[i].set_title(f"{feature}", fontweight="bold", fontsize=11)
            axes[i].set_xlabel("")
            axes[i].legend(
                fontsize=15,
            )

            # Add statistical comparison
            if len(signal_data) > 0 and len(background_data) > 0:
                t_stat, p_value = stats.ttest_ind(signal_data, background_data, equal_var=False)
                axes[i].text(
                    0.75,
                    0.15,
                    f"p-value: {p_value:.2e}",
                    transform=axes[i].transAxes,
                    fontsize=12,
                    verticalalignment="top",
                    bbox=dict(facecolor="white", alpha=0.8),
                )

        # Hide empty subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.suptitle(
            "Signal vs Background Feature Distributions",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
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
        figsize: tuple = (12, 10),
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot a correlation map only for the top number of features
        """
        # Compute correlation map
        corr = self._compute_correlation(df)

        # Set target columns
        target_corrs = corr["signal"].sort_values(ascending=False)
        top_features = target_corrs.head(num_features).index

        corr_top = corr.loc[top_features, top_features]

        # Create subplots
        fig, ax = plt.subplots(figsize=figsize)

        # Create mask only if requested
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr_top, dtype=bool))

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

        plt.title(f"Top {num_features} Features Correlated with Target", fontsize=14, pad=20)
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
        figsize: tuple = (20, 18),
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Plot full correlation map
        """
        # Create subplots
        fig, ax = plt.subplots(figsize=figsize)

        # Compute correlation map
        corr = self._compute_correlation(df)

        # Create mask only if requested
        mask = None
        if mask_upper:
            mask = np.triu(np.ones_like(corr, dtype=bool))

        # Optimize the large heatmap
        sns.heatmap(
            corr,
            annot=False,
            fmt=".2f",
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

    def _compute_correlation(
        self,
        df: pd.DataFrame,
    ):
        """Compute Pearson correlation for the features"""
        self.corr = df.corr()
        return self.corr

    def create_top_feature_pairplot(
        self,
        df: pd.DataFrame,
        num_features: int = 8,
        figsize: tuple = (20, 15),
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create pairplot of top features
        """
        # Compute correlations
        corr = self._compute_correlation(df)

        # Get top features correlated with target
        top_features = (
            corr["signal"].abs().sort_values(ascending=False).head(num_features).index.tolist()
        )

        # Pairplot of only important features
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

    def create_correlation_barplot(
        self,
        df: pd.DataFrame,
        figsize: tuple = (12, 10),
        num_features: int = 20,
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create barplot for the most important number of features selected
        """
        fig, ax = plt.subplots(figsize=figsize)
        corr = self._compute_correlation(df)
        corr_with_target = (
            corr["signal"].drop("signal", errors="ignore").sort_values(ascending=False)
        )

        # Take top and bottom features for balanced view
        top_pos = corr_with_target.head(num_features // 2)
        top_neg = corr_with_target.tail(num_features // 2)
        combined = pd.concat([top_pos, top_neg])

        combined.plot(
            kind="barh", ax=ax, color=combined.map(lambda x: "blue" if x >= 0 else "red")
        )
        plt.title(f"Top {num_features} Feature Correlations with Signal")
        plt.xlabel("Correlation Coefficient")
        fig.tight_layout()
        ax.grid(axis="x", alpha=0.3)

        # Save plot if save_config provided
        if save_config is not None:
            self.save_plot(fig, save_config, f"correlation_barplot_{num_features}_features")

        return fig

    def create_target_plot(
        self,
        df: pd.DataFrame,
        column: str = "signal",
        hue: str = "signal",
        figsize: tuple = (12, 10),
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create a simple one-dimensional plot
        """
        fig, ax = plt.subplots(figsize=figsize)
        sns.countplot(data=df, x=column, hue=hue, **kwargs)
        ax.set(
            xlabel="",
            ylabel="Percent (%)",
        )
        ax.xaxis.set_ticklabels([r"Background $\nu_{\mu}$", r"Neutrino $\nu_e$"])
        fig.tight_layout()
        ax.grid()

        # Save plot if save_config provided
        if save_config is not None:
            self.save_plot(fig, save_config, f"target_plot_{column}_{hue}")

        return fig

    def _update_config_from_kwargs(self, config, kwargs: Dict[str, Any]):
        """Update config object with kwargs"""
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def _format_stats_text(self, data: pd.Series) -> str:
        """Helper method to format statistics text"""
        return f"""N: {len(data):,}
Mean: {data.mean():.2f}
Std: {data.std():.2f}
Min: {data.min():.2f}
Max: {data.max():.2f}
Skew: {data.skew():.2f}"""
