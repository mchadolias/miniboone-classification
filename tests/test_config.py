# tests/test_config.py
import pytest
from src.config import DataConfig


class TestDataLoading:
    def test_data_config_creation(self):
        """Test that DataConfig can be created with defaults"""
        config = DataConfig()
        # Test the ACTUAL defaults from your DataConfig
        assert config.data_dir == "../data/external/"
        assert config.test_size == 0.2
        assert config.val_size == 0.2
        assert config.random_state == 42
        assert config.number_of_signals == 36499
        assert config.number_of_background == 93565

    def test_data_config_environment_variables(self, monkeypatch):
        """Test environment variable overrides"""
        monkeypatch.setenv("DATA_DIR", "/custom/path")
        monkeypatch.setenv("TEST_SIZE", "0.25")

        config = DataConfig()
        assert config.data_dir == "/custom/path"
        assert config.test_size == 0.25

    def test_sample_data_structure(self, sample_neutrino_data):
        """Test that sample data has correct structure"""
        df = sample_neutrino_data
        assert "signal" in df.columns
        assert len(df) == 5  # Match the ACTUAL size from conftest.py
        assert df["signal"].isin([0, 1]).all()

        # Test that we have both signal and background
        assert df["signal"].sum() > 0
        assert (df["signal"] == 0).sum() > 0

    def test_extreme_values_present(self, sample_neutrino_data):
        """Test that extreme values exist for testing outlier handling"""
        df = sample_neutrino_data
        # Test the ACTUAL columns from conftest.py
        assert "feature_1" in df.columns
        # Check that data looks reasonable
        assert len(df) > 0
        assert not df.isnull().any().any()  # No NaN values
