"""
Tests for the NeutrinoPlotter class.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock

from src.visualization.plotter import NeutrinoPlotter, setup_scientific_plotting
from src.config import BoxplotConfig, ViolinPlotConfig, SaveConfig


class TestNeutrinoPlotter:
    """Test suite for NeutrinoPlotter class."""

    @pytest.fixture
    def plotter(self):
        """Create a NeutrinoPlotter instance for testing."""
        return NeutrinoPlotter(style="seaborn")  # Use seaborn for consistent testing

    @pytest.fixture
    def save_config(self):
        """Create a SaveConfig for testing."""
        return SaveConfig(save_dir="tests/output", formats=["png"], dpi=100)

    def test_initialization(self, plotter):
        """Test that plotter initializes correctly."""
        assert plotter is not None
        assert plotter.style == "seaborn"
        assert plotter.corr is None

    def test_initialization_different_styles(self):
        """Test initialization with different styles."""
        # Test with science style (if available)
        with patch("src.visualization.plotter.SCIENCEPLOTS_AVAILABLE", True):
            plotter = NeutrinoPlotter(style="science")
            assert plotter.style == "science"

        # Test with default style
        plotter = NeutrinoPlotter(style="default")
        assert plotter.style == "default"

    def test_calculate_smart_limits(self, plotter, sample_neutrino_data):
        """Test the calculate_smart_limits static method."""
        features = ["feature_1", "feature_2"]
        limits = plotter.calculate_smart_limits(sample_neutrino_data, features, percentile=5.0)

        assert isinstance(limits, dict)
        assert "feature_1" in limits
        assert "feature_2" in limits

        # Check that limits are reasonable
        for feature, (lower, upper) in limits.items():
            assert lower < upper
            assert lower <= sample_neutrino_data[feature].max()
            assert upper >= sample_neutrino_data[feature].min()

    def test_calculate_smart_limits_empty_data(self, plotter):
        """Test smart limits with empty DataFrame."""
        empty_df = pd.DataFrame()
        features = ["feature_1"]

        with pytest.raises(KeyError):
            plotter.calculate_smart_limits(empty_df, features)

    # def test_create_violin_boxplot_combo(self, plotter, sample_neutrino_data, save_config):
    #     """Test violin plot creation."""
    #     fig = plotter.create_violin_boxplot_combo(
    #         sample_neutrino_data, figsize=(12, 8), save_config=save_config
    #     )

    #     assert isinstance(fig, plt.Figure)
    #     plt.close(fig)

    def test_create_pairplot(self, plotter, sample_neutrino_data, save_config=save_config):
        """Test pairplot creation."""
        fig = plotter.create_top_feature_pairplot(sample_neutrino_data, num_features=4)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_horizontal_boxplot_with_density(
        self, plotter, sample_neutrino_data, save_config
    ):
        """Test horizontal boxplot creation."""
        fig = plotter.create_horizontal_boxplot_with_density(
            sample_neutrino_data, figsize=(10, 8), save_config=save_config
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) > 0

        # Clean up
        plt.close(fig)

    def test_create_horizontal_boxplot_with_outlier_handling(self, plotter, sample_neutrino_data):
        """Test boxplot with different outlier handling methods."""
        # Test with outlier removal
        fig1 = plotter.create_horizontal_boxplot_with_density(
            sample_neutrino_data, outlier_handling="remove", outlier_percentile=5.0
        )

        # Test with outlier clipping
        fig2 = plotter.create_horizontal_boxplot_with_density(
            sample_neutrino_data, outlier_handling="clip", outlier_percentile=2.5
        )

        # Test without outlier handling
        fig3 = plotter.create_horizontal_boxplot_with_density(
            sample_neutrino_data, outlier_handling="none"
        )

        for fig in [fig1, fig2, fig3]:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_create_signal_vs_background_distributions(
        self, plotter, sample_neutrino_data, save_config
    ):
        """Test signal vs background distribution plots."""
        fig = plotter.create_signal_vs_background_distributions(
            sample_neutrino_data,
            features=["feature_1", "feature_2"],
            n_cols=2,
            save_config=save_config,
        )

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 2  # Should have at least 2 subplots

        plt.close(fig)

    def test_create_signal_vs_background_with_limits(self, plotter, sample_neutrino_data):
        """Test distribution plots with custom feature limits."""
        feature_limits = {"feature_1": (-2, 2), "feature_2": (0, 10)}

        fig = plotter.create_signal_vs_background_distributions(
            sample_neutrino_data,
            features=["feature_1", "feature_2"],
            feature_limits=feature_limits,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_create_correlation_plots(self, plotter, sample_neutrino_data, save_config):
        """Test correlation plot creation."""
        # Test top correlation map
        fig1 = plotter.create_top_correlation_map(
            sample_neutrino_data, num_features=5, save_config=save_config
        )

        # Test full correlation map
        fig2 = plotter.create_full_correlation_map(sample_neutrino_data, save_config=save_config)

        # Test correlation barplot
        fig3 = plotter.create_correlation_barplot(
            sample_neutrino_data, num_features=10, save_config=save_config
        )

        for fig in [fig1, fig2, fig3]:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_create_target_distribution_plot(self, plotter, sample_neutrino_data, save_config):
        """Test target distribution plot with different stats."""
        # Test with counts
        fig1 = plotter.create_target_distribution_plot(
            sample_neutrino_data, stat="count", save_config=save_config
        )

        # Test with percentages
        fig2 = plotter.create_target_distribution_plot(
            sample_neutrino_data, stat="percent", save_config=save_config
        )

        for fig in [fig1, fig2]:
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_plotter_with_empty_data(self, plotter):
        """Test plotter methods with empty DataFrame."""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError):
            plotter.create_horizontal_boxplot_with_density(empty_df)

    def test_plotter_with_missing_signal_column(self, plotter, sample_neutrino_data):
        """Test methods that require signal column with data missing signal."""
        data_no_signal = sample_neutrino_data.drop("signal", axis=1)

        with pytest.raises(ValueError):
            plotter.create_signal_vs_background_distributions(data_no_signal)

    @patch("os.makedirs")
    def test_save_plot_handles_directory_errors(self, mock_makedirs, plotter):
        """Test that save_plot handles directory creation errors"""
        save_config = SaveConfig(save_dir="/protected/path", formats=["png"])

        # Make directory creation fail
        mock_makedirs.side_effect = PermissionError("No permission")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Should not crash, just print error
        plotter.save_plot(fig, save_config, "test_save")

        # Should still attempt to create directory
        mock_makedirs.assert_called_once()

        plt.close(fig)

    @patch("os.makedirs")  # Mock directory creation to avoid permission issues
    @patch("matplotlib.figure.Figure.savefig")  # Mock the actual savefig method being called
    def test_save_plot_calls_savefig(
        self, mock_savefig, mock_makedirs, plotter, sample_neutrino_data
    ):
        """Test that save_plot method calls matplotlib's savefig correctly"""
        save_config = SaveConfig(save_dir="/fake/path", formats=["png"])

        # Create the figure first (this doesn't trigger save yet)
        fig = plotter.create_target_distribution_plot(sample_neutrino_data)

        # Now call save_plot with the mocked dependencies
        plotter.save_plot(fig, save_config, "test_plot")

        # Verify directory creation was attempted
        mock_makedirs.assert_called_once_with("/fake/path", exist_ok=True)

        # Verify savefig was called
        mock_savefig.assert_called_once()

        # Check the call arguments
        call_args = mock_savefig.call_args
        save_path = call_args[0][0]  # First positional argument
        assert "test_plot.png" in save_path

        # Check keyword arguments
        assert call_args[1]["dpi"] == 300  # From SaveConfig default
        assert call_args[1]["bbox_inches"] == "tight"

        plt.close(fig)

    @patch("os.makedirs")
    @patch("matplotlib.figure.Figure.savefig")
    def test_save_plot_handles_save_errors(self, mock_savefig, mock_makedirs, plotter):
        """Test that save_plot handles savefig errors gracefully"""
        save_config = SaveConfig(save_dir="/test/path", formats=["png", "pdf"])

        # Make savefig fail
        mock_savefig.side_effect = Exception("Disk full")

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Should not raise, just print error messages
        plotter.save_plot(fig, save_config, "test_save")

        # Should still attempt to save both formats
        assert mock_savefig.call_count == 2

        plt.close(fig)

    def test_auto_detect_scale_method(self, plotter):
        """Test the auto scale detection method."""
        # Test with positive-only data
        positive_data = pd.Series(np.random.lognormal(0, 2, 100))
        assert plotter._auto_detect_scale(positive_data) in ["linear", "log"]

        # Test with mixed positive/negative data
        mixed_data = pd.Series(np.random.normal(0, 10, 100))
        assert plotter._auto_detect_scale(mixed_data) in ["linear", "symlog"]

        # Test with empty data
        empty_data = pd.Series([])
        assert plotter._auto_detect_scale(empty_data) == "linear"

    def test_handle_outliers_method(self, plotter):
        """Test outlier handling method."""
        data = pd.Series([1, 2, 3, 100, 200])  # With outliers

        # Test removal
        cleaned_data, info = plotter._handle_outliers(data, method="remove", percentile=10)
        assert len(cleaned_data) < len(data)
        assert info["outliers_removed"] > 0

        # Test clipping
        clipped_data, info = plotter._handle_outliers(data, method="clip", percentile=10)
        assert len(clipped_data) == len(data)
        assert info["outliers_removed"] > 0

        # Test no handling
        original_data, info = plotter._handle_outliers(data, method="none")
        assert len(original_data) == len(data)
        assert info["outliers_removed"] == 0

    @patch("matplotlib.pyplot.style.use")
    def test_setup_scientific_plotting(self, mock_style_use):
        """Test the setup_scientific_plotting function."""
        # Test with scienceplots available
        with patch("src.visualization.plotter.SCIENCEPLOTS_AVAILABLE", True):
            with patch("src.visualization.plotter.check_latex_available", return_value=True):
                setup_scientific_plotting(style="science")
                mock_style_use.assert_called_with(["science", "ieee", "grid"])

        # Test with seaborn
        setup_scientific_plotting(style="seaborn")
        mock_style_use.assert_called_with("seaborn-v0_8-whitegrid")

    def test_plotter_with_large_dataset(self, plotter):
        """Test plotter performance with larger dataset."""
        # Create larger dataset
        np.random.seed(42)
        large_data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 1000),
                "feature_2": np.random.exponential(1, 1000),
                "signal": np.random.choice([0, 1], 1000, p=[0.8, 0.2]),
            }
        )

        # Should complete without errors
        fig = plotter.create_horizontal_boxplot_with_density(
            large_data, max_features=10  # Limit for performance
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotterEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def plotter(self):
        return NeutrinoPlotter()

    def test_invalid_preset(self, plotter, sample_neutrino_data):
        """Test using invalid preset names."""
        with pytest.raises(ValueError):
            plotter.create_horizontal_boxplot_with_density(
                sample_neutrino_data, preset="invalid_preset_name"
            )

    def test_invalid_outlier_method(self, plotter, sample_neutrino_data):
        """Test using invalid outlier handling method."""
        fig = plotter.create_horizontal_boxplot_with_density(
            sample_neutrino_data, outlier_handling="invalid_method"
        )
        # Should fall back to default without crashing
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_feature_data(self, plotter):
        """Test with single feature DataFrame."""
        single_feature_data = pd.DataFrame(
            {"feature_1": [1, 2, 3, 4, 5], "signal": [0, 1, 0, 1, 0]}
        )

        fig = plotter.create_signal_vs_background_distributions(
            single_feature_data, features=["feature_1"]
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_constant_feature_data(self, plotter):
        """Test with constant (zero variance) feature."""
        constant_data = pd.DataFrame(
            {"constant_feature": [1, 1, 1, 1, 1], "signal": [0, 1, 0, 1, 0]}  # No variance
        )

        with pytest.raises(ValueError):
            plotter.create_signal_vs_background_distributions(
                constant_data, features=["constant_feature"]
            )

    def test_feature_data(self, plotter):
        """Test with flactuating data feature."""
        flactuating_data = pd.DataFrame(
            {"feature": [1, 3, 5, 3, 6], "signal": [0, 1, 0, 1, 0]}  # Variance
        )

        fig = plotter.create_signal_vs_background_distributions(
            flactuating_data, features=["feature"]
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
