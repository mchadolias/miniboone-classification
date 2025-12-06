"""
Unit tests for the PhysicsPlotter class.
Covers PCA, t-SNE, and feature summary visualizations.
"""

import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch

from src.plotter import DimensionalityReductionPlotter
from src.config import SaveConfig


@pytest.fixture
def plotter():
    """Return a PhysicsPlotter instance."""
    return DimensionalityReductionPlotter(style="science")


@pytest.fixture
def save_config(tmp_path):
    """Provide temporary save configuration."""
    return SaveConfig(save_dir=tmp_path, formats=["png"], dpi=100)


# -------------------------------------------------------------------------
# PCA Scatter
# -------------------------------------------------------------------------


def test_plot_pca_scatter(plotter, sample_neutrino_data, save_config):
    """Test PCA scatter creation."""
    fig = plotter.plot_pca_scatter(
        df=sample_neutrino_data,
        target="signal",
        n_components=2,
        save_config=save_config,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_pca_invalid_target(plotter, sample_neutrino_data):
    """Should raise if target missing."""
    invalid_df = sample_neutrino_data.drop(columns=["signal"])
    with pytest.raises(ValueError):
        plotter.plot_pca_scatter(invalid_df)


# -------------------------------------------------------------------------
# t-SNE Embedding
# -------------------------------------------------------------------------


@patch("src.plotter.dimensionality_reduction_plotter.TSNE.fit_transform")
def test_plot_tsne_embedding(mock_fit_transform, plotter, sample_neutrino_data, save_config):
    """Test t-SNE embedding generation with direct fit_transform patch."""
    mock_fit_transform.return_value = sample_neutrino_data[["feature_1", "feature_2"]].values

    fig = plotter.plot_tsne_embedding(
        df=sample_neutrino_data,
        target="signal",
        perplexity=30,
        max_iter=250,
        save_config=save_config,
    )

    mock_fit_transform.assert_called_once()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_tsne_invalid_target(plotter, synthetic_miniboone_data):
    """Should raise if target column missing."""
    df_invalid = synthetic_miniboone_data.drop(columns=["signal"])
    with pytest.raises(ValueError):
        plotter.plot_tsne_embedding(df_invalid)
