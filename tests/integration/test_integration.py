"""
Integration tests for the complete data pipeline.
"""

import pytest
import pandas as pd
from src.data.data_handler import MiniBooNEDataHandler
from src.visualization.plotter import NeutrinoPlotter
from src.config import DataConfig


class TestIntegration:
    """Test complete workflow integration."""

    def test_complete_pipeline_with_mock_data(self, sample_neutrino_data, tmp_path):
        """Test complete pipeline with mocked data."""
        # Setup
        config = DataConfig(data_dir=str(tmp_path))
        handler = MiniBooNEDataHandler(config=config)
        plotter = NeutrinoPlotter()

        # Mock the data loading
        handler.df = sample_neutrino_data

        # Test full workflow
        handler.clean_data()
        splits = handler.preprocess()

        # Verify splits
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits

        # Test plotting on processed data
        fig = plotter.create_target_distribution_plot(handler.df)
        assert fig is not None

    def test_plotter_with_processed_splits(self, data_handler_with_loaded_data):
        """Test plotter functionality with preprocessed data."""
        handler = data_handler_with_loaded_data
        plotter = NeutrinoPlotter()

        splits = handler.preprocess()

        # Create DataFrame from training split for plotting
        X_train, y_train = splits["train"]
        train_df = pd.DataFrame(X_train, columns=handler.get_feature_names())
        train_df["signal"] = y_train.values

        # Test various plots
        fig1 = plotter.create_target_distribution_plot(train_df)
        fig2 = plotter.create_signal_vs_background_distributions(
            train_df, features=handler.get_feature_names()[:3]  # Limit for performance
        )

        assert fig1 is not None
        assert fig2 is not None
