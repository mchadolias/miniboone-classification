from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import SaveConfig
from src.plotter import ScientificPlotter
from src.stats.statistical_analysis import (
    compute_auc_score,
    compute_bootstrap_error,
    compute_effect_size,
    compute_js_divergence,
    compute_ks_pvalue,
    compute_mannwhitney_pvalue,
)
from src.utils.logger import get_global_logger

logger = get_global_logger(__name__)


class NeutrinoPlotter(ScientificPlotter):
    """
    Physics-aware visualization class for signal vs. background analysis
    in neutrino datasets (e.g., MiniBooNE).

    This class provides plotting tools to visualize feature separation,
    correlations, and target distributions for physics-inspired
    classification problems.

    Methods
    -------
    plot_feature_separation
        Plot KDE or histogram overlays for signal vs. background distributions.
    plot_target_distribution
        Plot target distribution with bootstrapped error bars.
    plot_top_correlation_heatmap
        Plot a heatmap of top correlated features with the target or per class.
    plot_top_correlations
        Plot the top N features most correlated with the target variable.

    Attributes
    ----------
    style : str
        Matplotlib/SciencePlots style preset for figures.

    Examples
    --------
    >>> from src.plotter import NeutrinoPlotter
    >>> plotter = NeutrinoPlotter(style="science")
    >>> fig = plotter.plot_feature_separation(df, features=["col_0", "col_1"], target="signal")
    >>> fig.show()
    """

    # -------------------------------------------------------------------------
    # Enhanced Feature Separation Plot
    # -------------------------------------------------------------------------
    def plot_feature_separation(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str = "signal",
        save_config: Optional[SaveConfig] = None,
        x_axis_trim_percentiles: Tuple[float, float] = (1, 99),
        show_mean_median: bool = False,
        plot_as_histogram: bool = False,
        annotate_stats: bool = True,
        stat_method: Literal["mannwhitney", "ks", "ttest"] = "mannwhitney",
    ) -> plt.Figure:
        """
        Plot KDE or histogram overlays for signal vs. background distributions,
        optionally annotated with statistical separation metrics.

        Args:
            df (pd.DataFrame): Input dataset with features and target column.
            features (List[str]): Columns to visualize.
            target (str): Target column (e.g., 'signal').
            save_config (Optional[SaveConfig]): Optional save configuration.
            x_axis_trim_percentiles (Tuple[float, float]): Percentiles for x-axis trimming.
            show_mean_median (bool): Whether to show mean/median lines.
            plot_as_histogram (bool): If True, plot histograms instead of KDE.
            annotate_stats (bool): If True, annotate with AUC, p-value, effect size, etc.
            stat_method (Literal["mannwhitney", "ks", "ttest"]): Statistical test type.

        Returns:
            plt.Figure: The resulting figure.
        """
        self.validate_dataframe(df, required_cols=[target])

        n_cols = min(3, len(features))
        n_rows = (len(features) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

        # Store metrics for optional export
        summary = []

        for i, feature in enumerate(features):
            ax = axes[i]
            try:
                bg = df[df[target] == 0][feature].dropna()
                sig = df[df[target] == 1][feature].dropna()
                combined = pd.concat([bg, sig])

                # Skip constant columns
                if combined.nunique() < 2:
                    ax.set_visible(False)
                    continue

                # Trim x-axis extremes
                lower, upper = np.percentile(combined, x_axis_trim_percentiles)
                ax.set_xlim(lower, upper)

                # Choose visualization type
                if plot_as_histogram:
                    ax.hist(
                        bg, bins=50, alpha=0.5, label="Background", color="black", density=True
                    )
                    ax.hist(sig, bins=50, alpha=0.5, label="Signal", color="red", density=True)
                else:
                    sns.kdeplot(bg, ax=ax, label="Background", fill=True, alpha=0.5, color="black")
                    sns.kdeplot(sig, ax=ax, label="Signal", fill=True, alpha=0.5, color="red")

                # Compute statistics if enabled
                auc = compute_auc_score(df, feature, target)
                eff = compute_effect_size(bg, sig)
                if stat_method == "mannwhitney":
                    pval = compute_mannwhitney_pvalue(bg, sig)
                elif stat_method == "ks":
                    pval = compute_ks_pvalue(bg, sig)
                else:
                    pval = np.nan
                jsd = compute_js_divergence(bg.values, sig.values)

                # Save summary entry
                summary.append(
                    {
                        "feature": feature,
                        "auc": auc,
                        "effect_size": eff,
                        "p_value": pval,
                        "js_divergence": jsd,
                    }
                )

                # Add annotations on the plot
                if annotate_stats:
                    stars = (
                        "***"
                        if pval < 0.001
                        else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    )
                    ax.text(
                        0.98,
                        0.95,
                        f"AUC={auc:.2f}\nCohen’s d={eff:.2f}\nJSD={jsd:.2f}\nP={pval:.2e} {stars}",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
                    )

                if show_mean_median:
                    for data, color, label in [(bg, "black", "BG"), (sig, "red", "SIG")]:
                        ax.axvline(
                            data.mean(), linestyle="--", color=color, linewidth=1, alpha=0.7
                        )
                        ax.axvline(
                            data.median(), linestyle=":", color=color, linewidth=1, alpha=0.7
                        )

                ax.legend()

            except Exception as e:
                logger.warning(f"Could not plot separation for '{feature}': {e}")
                ax.set_visible(False)

        # Hide unused axes
        for j in range(len(features), len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()

        # Save figure and optional metrics summary
        if save_config:
            self.export_figure(fig, save_config, "feature_separation")
            summary_df = pd.DataFrame(summary)
            summary_csv = save_config.save_dir / "feature_separation_stats.csv"
            summary_df.to_csv(summary_csv, index=False)
            logger.info(f"Saved feature separation summary: {summary_csv}")

        return fig

    # -------------------------------------------------------------------------
    # Target Distribution Plot
    # -------------------------------------------------------------------------
    def plot_target_distribution(
        self,
        df: pd.DataFrame,
        column: str = "signal",
        stat: str = "count",
        show_legend: bool = False,
        n_boot: int = 1000,
        save_config: Optional[SaveConfig] = None,
        **kwargs,
    ) -> plt.Figure:
        """Plot target distribution with bootstrapped error bars.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        column : str
            Column containing binary target values.
        stat : str
            Statistic to plot ('count' or 'percent').
        show_legend : bool
            Whether to display a custom legend.
        n_boot : int
            Number of bootstrap resamples for error estimation.
        save_config : Optional[SaveConfig]
            Configuration for saving the plot.
        **kwargs
            Additional matplotlib keyword arguments.

        Returns
        -------
        plt.Figure
            Matplotlib figure object.
        """
        self.validate_dataframe(df, required_cols=[column])

        plot_df = df.copy()
        unique_vals = sorted(plot_df[column].dropna().unique())

        if len(unique_vals) != 2:
            raise ValueError(f"Target column '{column}' must have exactly 2 unique values.")

        label_map = {
            unique_vals[0]: r"Background ($\nu_\mu$)",
            unique_vals[1]: r"Signal ($\nu_e$)",
        }
        plot_df["label"] = plot_df[column].map(label_map)
        order = list(label_map.values())

        counts = plot_df["label"].value_counts(normalize=(stat == "percent"))
        errors = {
            label: compute_bootstrap_error(plot_df["label"], label, n_boot, stat)
            for label in order
        }

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))

        # Bar chart with error bars
        bar_container = ax.bar(
            order,
            [counts.get(label, 0) for label in order],
            yerr=[errors.get(label, 0) for label in order],
            capsize=6,
            color=["black", "red"],
            alpha=0.8,
            edgecolor="k",
        )

        total = len(plot_df)
        for bar in bar_container:
            height = bar.get_height()
            if stat == "percent":
                annotation = f"{height:.2f}%"
            else:
                annotation = f"{int(height):,} ({(height / total) * 100:.1f}\\%)"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(ax.get_ylim()) * 0.01,
                annotation,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_ylabel("Percent (\\%)" if stat == "percent" else "Count")
        ax.set_xlabel("")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.spines[["top", "right"]].set_visible(False)

        if show_legend:
            legend = [
                plt.Rectangle((0, 0), 1, 1, facecolor="black", label=r"Background ($\nu_\mu$)"),
                plt.Rectangle((0, 0), 1, 1, facecolor="red", label=r"Signal ($\nu_e$)"),
            ]
            ax.legend(handles=legend, loc="upper right")

        fig.tight_layout()

        if save_config:
            self.export_figure(fig, save_config, "target_distribution")

        return fig

    # -------------------------------------------------------------------------
    # Correlation Map (Per Class)
    # -------------------------------------------------------------------------
    def plot_top_correlation_heatmap(
        self,
        df: pd.DataFrame,
        target: str = "signal",
        top_n: int = 10,
        method: str = "pearson",
        positive_only: bool = False,
        negative_only: bool = False,
        per_class: bool = False,
        save_config: Optional[SaveConfig] = None,
    ) -> Dict[str, plt.Figure]:
        """Plot a heatmap of top correlated features with the target or per class.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.
        target : str
            Target column.
        top_n : int
            Number of top correlated features to display.
        method : str
            Correlation method ('pearson', 'spearman', etc.).
        positive_only : bool
            If True, show only positively correlated features.
        negative_only : bool
            If True, show only negatively correlated features.
        per_class : bool
            If True, plot separate heatmaps for signal and background.
        save_config : Optional[SaveConfig]
            Configuration for saving figures.

        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary with figure handles, keyed by class label.
        """
        self.validate_dataframe(df, required_cols=[target])

        def _make_heatmap(data: pd.DataFrame, label: str) -> plt.Figure:
            corr = data.corr(method=method)
            target_corr = corr[target].drop(index=target)

            if positive_only:
                target_corr = target_corr[target_corr > 0]
            elif negative_only:
                target_corr = target_corr[target_corr < 0]

            top_feats = target_corr.abs().nlargest(top_n).index.tolist()
            top_feats.append(target)

            fig, ax = plt.subplots()
            sns.heatmap(
                corr.loc[top_feats, top_feats],
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax,
            )
            ax.set_title(f"Top {top_n} Correlations with {target} ({label})")
            fig.tight_layout()

            if save_config:
                self.export_figure(fig, save_config, f"correlation_map_{label}")

            return fig

        if per_class:
            figs = {}
            for val in sorted(df[target].dropna().unique()):
                label = "Signal" if val == 1 else "Background"
                subset = df[df[target] == val]
                figs[label] = _make_heatmap(subset, label)
            return figs
        else:
            return {"Combined": _make_heatmap(df, "Combined")}

    # -------------------------------------------------------------------------
    # Top Feature Correlations with Target
    # -------------------------------------------------------------------------
    def plot_top_correlations(
        self,
        df: pd.DataFrame,
        target: str = "signal",
        top_n: int = 10,
        method: str = "pearson",
        positive_only: bool = False,
        negative_only: bool = False,
        per_class: bool = False,
        show_values: bool = True,
        save_config: Optional[SaveConfig] = None,
    ) -> dict:
        """
        Plot the top N features most correlated with the target variable.

        Supports filtering by positive/negative correlations and optional
        per-class breakdowns (signal vs background).

        Args:
            df (pd.DataFrame): Input DataFrame containing features and target.
            target (str): Target column name (default: 'signal').
            top_n (int): Number of features to display.
            method (str): Correlation method ('pearson', 'spearman', 'kendall').
            positive_only (bool): Show only positively correlated features.
            negative_only (bool): Show only negatively correlated features.
            per_class (bool): Generate separate correlation plots for each class.
            show_values (bool): Whether to display correlation coefficient text on bars.
            save_config (Optional[SaveConfig]): Optional configuration for saving.

        Returns:
            dict: Dictionary mapping {label: matplotlib.Figure}.
        """
        self.validate_dataframe(df, required_cols=[target])
        numeric_df = df.select_dtypes(include=[np.number])

        if target not in numeric_df.columns:
            raise ValueError(f"Target column '{target}' must be numeric for correlation.")

        def _make_barplot(data: pd.DataFrame, label: str) -> plt.Figure:
            corr = data.corr(method=method)[target].drop(target)

            if positive_only:
                corr = corr[corr > 0]
            elif negative_only:
                corr = corr[corr < 0]

            top_corr = corr.abs().sort_values(ascending=False).head(top_n)
            top_corr = corr[top_corr.index]  # preserve sign
            colors = ["blue" if val > 0 else "red" for val in top_corr]

            fig, ax = plt.subplots()
            bars = ax.barh(top_corr.index, top_corr.values, color=colors, alpha=0.8)

            ax.set_title(f"Top {top_n} {method.title()} Correlated Features ({label})")
            ax.set_xlabel(f"{method.title()} Correlation Coefficient")
            ax.set_ylabel("Feature")
            ax.invert_yaxis()
            ax.grid(True, axis="x", linestyle="--", alpha=0.4)

            # Annotate correlation values
            if show_values:
                for bar, val in zip(bars, top_corr.values):
                    ax.text(
                        val + 0.02 * np.sign(val),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:.2f}",
                        va="center",
                        ha="left" if val > 0 else "right",
                        fontsize=9,
                        fontweight="bold",
                        color="black",
                    )

            fig.tight_layout()

            if save_config:
                label_clean = label.lower().replace(" ", "_")
                filename = f"top_{top_n}_correlations_{label_clean}"
                self.export_figure(fig, save_config, filename)

            return fig

        # -----------------------------------------------------------------
        # Combined correlation plot
        # -----------------------------------------------------------------
        figures = {}
        if per_class:
            for val in sorted(df[target].dropna().unique()):
                label = "Signal" if val == 1 else "Background"
                subset = df[df[target] == val]
                figures[label] = _make_barplot(subset, label)
        else:
            figures["Combined"] = _make_barplot(df, "Combined")

        return figures
