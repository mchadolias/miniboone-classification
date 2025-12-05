"""
Unit tests for the ScientificPlotter base class.
Covers style setup, dataframe validation, and save logic.
"""

from pathlib import Path
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import pandas as pd

from src.plotter import ScientificPlotter
from src.config import SaveConfig


@pytest.fixture
def base_plotter():
    """Return a ScientificPlotter instance."""
    return ScientificPlotter(style="science")


@pytest.fixture
def save_config(
    tmp_path: Path,
) -> SaveConfig:
    """Return a SaveConfig instance with a temporary directory."""
    return SaveConfig(save_dir=tmp_path, formats=["png"], dpi=200)


# -------------------------------------------------------------------------
# Style and Setup
# -------------------------------------------------------------------------


def test_plot_style_science(base_plotter):
    """Ensure 'science' style initializes correctly."""
    assert base_plotter.style == "science"


@patch("matplotlib.pyplot.style.use")
def test_plot_style_invalid(mock_style_use):
    """Ensure invalid style falls back to default."""
    plotter = ScientificPlotter(style="invalid_style")
    mock_style_use.assert_called()


# -------------------------------------------------------------------------
# Data Validation
# -------------------------------------------------------------------------


def test_validate_dataframe_valid(base_plotter):
    """Validation passes for correct DataFrame."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    base_plotter.validate_dataframe(df, required_cols=["a", "b"])  # no exception


def test_validate_dataframe_missing_column(base_plotter):
    """Should raise error when required columns missing."""
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError):
        base_plotter.validate_dataframe(df, required_cols=["a", "b"])


def test_validate_dataframe_empty(base_plotter):
    """Should raise for empty DataFrame."""
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        base_plotter.validate_dataframe(df)


# -------------------------------------------------------------------------
# Save Plot Logic
# -------------------------------------------------------------------------


@patch("matplotlib.figure.Figure.savefig")
def test_save_plot_saves_correctly(mock_savefig, base_plotter, save_config):
    """Ensure figure is saved to correct location."""
    fig, _ = plt.subplots()
    base_plotter.export_figure(fig, save_config, filename="test_plot")
    mock_savefig.assert_called()
    plt.close(fig)


def test_save_plot_without_config_raises(base_plotter):
    """Should raise if save_config is None."""
    fig, _ = plt.subplots()
    with pytest.raises(ValueError):
        base_plotter.export_figure(fig, None, filename="test_plot")
    plt.close(fig)
