"""
Unit tests for the NeutrinoPlotter class.

These tests validate feature separation plots, correlation visualizations,
and statistical annotations using synthetic MiniBooNE-like data.
"""

import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch
from src.plotter import NeutrinoPlotter
from src.config import SaveConfig
from pathlib import Path


@pytest.fixture
def plotter():
    """Return a NeutrinoPlotter instance with 'science' style."""
    return NeutrinoPlotter(style="science")


@pytest.fixture
def save_config(tmp_path):
    """Temporary save config for tests."""
    return SaveConfig(save_dir=tmp_path, formats=["png"], dpi=100)


# -------------------------------------------------------------------------
# Core Tests
# -------------------------------------------------------------------------


def test_plot_feature_separation_basic(plotter, synthetic_miniboone_data, save_config):
    """Test feature separation plots with statistical annotations."""
    fig = plotter.plot_feature_separation(
        df=synthetic_miniboone_data,
        features=["feature_1", "feature_2", "feature_3"],
        target="signal",
        annotate_stats=True,
        show_mean_median=True,
        save_config=save_config,
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_separation_handles_constant_feature(plotter, sample_neutrino_data):
    """Ensure constant features are skipped gracefully."""
    sample_neutrino_data["constant"] = 1.0
    fig = plotter.plot_feature_separation(
        df=sample_neutrino_data, features=["constant", "feature_1"], target="signal"
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_separation_invalid_target(plotter, synthetic_miniboone_data):
    """Should raise if target column missing."""
    invalid_df = synthetic_miniboone_data.drop(columns=["signal"])
    with pytest.raises(ValueError):
        plotter.plot_feature_separation(invalid_df, features=["feature_1", "feature_2"])


def test_plot_top_correlations_combined(plotter, synthetic_miniboone_data, save_config):
    """Test combined correlation barplot."""
    figs = plotter.plot_top_correlations(
        df=synthetic_miniboone_data,
        target="signal",
        top_n=8,
        method="pearson",
        per_class=False,
        save_config=save_config,
    )
    assert "Combined" in figs
    assert isinstance(figs["Combined"], plt.Figure)
    plt.close(figs["Combined"])


def test_plot_top_correlations_per_class(plotter, synthetic_miniboone_data):
    """Test correlation barplots for signal and background separately."""
    figs = plotter.plot_top_correlations(
        df=synthetic_miniboone_data,
        target="signal",
        top_n=5,
        per_class=True,
    )
    assert all(isinstance(f, plt.Figure) for f in figs.values())
    plt.close("all")


def test_invalid_correlation_target_non_numeric(plotter, synthetic_miniboone_data):
    """Should raise for non-numeric target."""
    synthetic_miniboone_data["signal"] = synthetic_miniboone_data["signal"].astype(str)
    with pytest.raises(ValueError):
        plotter.plot_top_correlations(synthetic_miniboone_data, target="signal")


# -------------------------------------------------------------------------
# Save Behavior Tests
# -------------------------------------------------------------------------


from unittest.mock import patch
from pathlib import Path


@patch("src.plotter.base_plotter.Path.mkdir")
@patch("matplotlib.figure.Figure.savefig")
def test_save_plot_called_correctly(
    mock_savefig, mock_makedirs, plotter, synthetic_miniboone_data, save_config
):
    """Ensure plot saving logic works and figures are exported correctly."""
    fig = plotter.plot_feature_separation(
        df=synthetic_miniboone_data,
        features=["feature_1"],
        target="signal",
    )
    plotter.export_figure(fig, save_config, "test_plot")

    assert mock_makedirs.call_count >= 1
    mock_makedirs.assert_any_call(parents=True, exist_ok=True)

    assert mock_savefig.call_count >= 1

    mock_savefig.assert_any_call(
        save_config.save_dir / "test_plot.png",
        dpi=save_config.dpi,
        bbox_inches=save_config.bbox_inches,
        facecolor="white",
        transparent=save_config.transparent,
    )


def test_plotter_handles_empty_dataframe(plotter):
    """Should raise error for empty DataFrame."""
    import pandas as pd

    with pytest.raises(ValueError):
        plotter.plot_feature_separation(pd.DataFrame(), features=["feature_1"])
