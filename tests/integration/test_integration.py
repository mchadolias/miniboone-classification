"""
Integration tests for the complete data pipeline.
"""

import pytest
import pandas as pd
from src.data.data_handler import MiniBooNEDataHandler
from src.plotter import NeutrinoPlotter
from src.config import DataConfig
from pathlib import Path


class TestIntegration:
    """Test complete workflow integration."""

    def test_complete_pipeline_with_mock_data(self, sample_neutrino_data, tmp_path):
        """Test complete pipeline with mocked data."""
        # Setup
        config = DataConfig(data_dir=Path(tmp_path))
        handler = MiniBooNEDataHandler(config=config)
        plotter = NeutrinoPlotter()

        # Mock the data loading
        handler.df = sample_neutrino_data

        # Test full workflow
        splits = handler.process()

        # Verify splits
        assert "train" in splits
        assert "test" in splits

        # Test plotting on processed data
        fig = plotter.plot_feature_separation(
            df=handler.df,
            features=handler.get_feature_names()[:5],  # Limit features for test speed
            target="signal",
        )
        assert fig is not None
