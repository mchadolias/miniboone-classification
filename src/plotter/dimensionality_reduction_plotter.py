import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from src.plotter import ScientificPlotter
from src.config import SaveConfig
from src.utils.logger import get_module_logger

logger = get_module_logger(__name__)


class DimensionalityReductionPlotter(ScientificPlotter):
    """
    Plotter for visualizing dimensionality reduction results (PCA, t-SNE, UMAP).

    This class provides standardized visualizations for feature embeddings,
    allowing for exploration of signal/background separability and latent
    structure in the dataset.

    Attributes
    ----------
    style : str
        Matplotlib/SciencePlots style preset for figures.
    """

    def __init__(self, style: str = "science") -> None:
        super().__init__(style)
        logger.info("DimensionalityReductionPlotter initialized")

    # -------------------------------------------------------------------------
    # PCA Scatter Plot
    # -------------------------------------------------------------------------
    def plot_pca_scatter(
        self,
        df: pd.DataFrame,
        target: str = "signal",
        n_components: int = 2,
        save_config: Optional[SaveConfig] = None,
    ) -> plt.Figure:
        """Plot a PCA projection of the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with features and target column.
        target : str
            Name of the target column.
        n_components : int
            Number of principal components to project.
        save_config : Optional[SaveConfig]
            Save configuration for exporting the figure.

        Returns
        -------
        plt.Figure
            PCA scatter plot.
        """
        self.validate_dataframe(df, required_cols=[target])

        features = df.drop(columns=[target])
        scaled = StandardScaler().fit_transform(features)
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(scaled)

        explained_var = np.sum(pca.explained_variance_ratio_) * 100
        labels = df[target].values

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            components[:, 0],
            components[:, 1],
            c=labels,
            cmap="coolwarm",
            alpha=0.6,
        )
        ax.set_title(f"PCA Scatter ({explained_var:.1f}% variance explained)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")

        legend_labels = ["Background", "Signal"]
        handles = [
            plt.Line2D(
                [0], [0], marker="o", color="w", label=label, markerfacecolor=col, markersize=8
            )
            for label, col in zip(legend_labels, ["blue", "red"])
        ]
        ax.legend(handles=handles)

        fig.tight_layout()
        if save_config:
            self.export_figure(fig, save_config, "pca_scatter")

        return fig

    # -------------------------------------------------------------------------
    # t-SNE Embedding
    # -------------------------------------------------------------------------
    def plot_tsne_embedding(
        self,
        df: pd.DataFrame,
        target: str = "signal",
        perplexity: int = 30,
        max_iter: int = 1000,
        save_config: Optional[SaveConfig] = None,
    ) -> plt.Figure:
        """Visualize t-SNE embeddings of the dataset.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        target : str
            Binary target column.
        perplexity : int
            t-SNE perplexity parameter.
        max_iter : int
            Number of optimization iterations.
        save_config : Optional[SaveConfig]
            Save configuration for exporting the figure.

        Returns
        -------
        plt.Figure
            t-SNE embedding plot.
        """
        self.validate_dataframe(df, required_cols=[target])

        X = df.drop(columns=[target]).values
        y = df[target].values
        X_scaled = StandardScaler().fit_transform(X)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            max_iter=max_iter,
            init="pca",
            random_state=42,
        )
        X_embedded = tsne.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=X_embedded[:, 0],
            y=X_embedded[:, 1],
            hue=y,
            palette="coolwarm",
            alpha=0.6,
            ax=ax,
        )
        ax.set_title(f"t-SNE Embedding (Perplexity={perplexity})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title=target)

        fig.tight_layout()
        if save_config:
            self.export_figure(fig, save_config, "tsne_embedding")

        return fig

    # -------------------------------------------------------------------------
    # UMAP Embedding (Optional)
    # -------------------------------------------------------------------------
    def plot_umap_embedding(
        self,
        df: pd.DataFrame,
        target: str = "signal",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        save_config: Optional[SaveConfig] = None,
    ) -> Optional[plt.Figure]:
        """Visualize UMAP embeddings if umap-learn is installed.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        target : str
            Binary target column.
        n_neighbors : int
            Number of neighbors used in local manifold approximation.
        min_dist : float
            Minimum distance between embedded points.
        save_config : Optional[SaveConfig]
            Save configuration for exporting the figure.

        Returns
        -------
        Optional[plt.Figure]
            UMAP embedding plot if UMAP is available, otherwise None.
        """
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not installed. Skipping UMAP plot.")
            return None

        self.validate_dataframe(df, required_cols=[target])

        X = df.drop(columns=[target]).values
        y = df[target].values
        X_scaled = StandardScaler().fit_transform(X)

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42,
        )
        embedding = reducer.fit_transform(X_scaled)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=embedding[:, 0],
            y=embedding[:, 1],
            hue=y,
            palette="coolwarm",
            alpha=0.6,
            ax=ax,
        )
        ax.set_title(f"UMAP Embedding (n_neighbors={n_neighbors}, min_dist={min_dist})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title=target)

        fig.tight_layout()
        if save_config:
            self.export_figure(fig, save_config, "umap_embedding")

        return fig
