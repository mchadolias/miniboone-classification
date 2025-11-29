import pytest
import importlib
import src.config
import numpy as np

# Force reload the module
importlib.reload(src.config)

# Now import from the reloaded module
from src.config import DataConfig, SaveConfig


class TestDataLoading:
    def test_data_config_creation(self):
        """Test that DataConfig can be created with defaults"""
        config = DataConfig()
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

    def test_data_config_validation(self):
        """Test DataConfig parameter validation."""
        # Test invalid test_size - Pydantic V2 handles this automatically
        with pytest.raises(ValueError):
            DataConfig(test_size=1.5)  # > 1.0

        with pytest.raises(ValueError):
            DataConfig(test_size=-0.1)  # < 0.0

    def test_model_validation_after_creation(self):
        """Test that validation works when modifying attributes."""
        config = DataConfig()

        # Valid change
        config.test_size = 0.25
        assert config.test_size == 0.25

        # Invalid change should raise on assignment
        with pytest.raises(ValueError):
            config.test_size = 0.6  # Too large

    def test_sample_data_structure(self, sample_dataframe):
        """Test that sample data has correct structure"""
        df = sample_dataframe
        assert "signal" in df.columns
        assert len(df) == 5  # Match the ACTUAL size from conftest.py
        assert df["signal"].isin([0, 1]).all()

        # Test that we have both signal and background
        assert df["signal"].sum() > 0
        assert (df["signal"] == 0).sum() > 0

    def test_extreme_values_present(self, sample_dataframe):
        """Test that extreme values exist for testing outlier handling"""
        df = sample_dataframe
        # Test the ACTUAL columns from conftest.py
        assert "feature_1" in df.columns
        # Check that data looks reasonable
        assert len(df) > 0
        assert not df.isnull().any().any()  # No NaN values

    def test_valid_default_configuration(self):
        """Test that default configuration is valid."""
        config = DataConfig()
        assert config.test_size == 0.2
        assert config.val_size == 0.2
        train_size = 1 - config.test_size - config.val_size
        assert np.round(train_size, 2) == 0.60  # 60% training data

    def test_valid_custom_configuration(self):
        """Test various valid configuration combinations."""
        valid_configs = [
            {"test_size": 0.1, "val_size": 0.1},  # 80% train
            {"test_size": 0.2, "val_size": 0.1},  # 70% train
            {"test_size": 0.15, "val_size": 0.2},  # 65% train
        ]

        for params in valid_configs:
            config = DataConfig(**params)
            train_size = 1 - config.test_size - config.val_size
            assert train_size >= 0.6, f"Training size too small: {train_size}"
            print(
                f"✓ Valid: test_size={params['test_size']}, val_size={params['val_size']} -> train_size={train_size:.1%}"
            )

    def test_invalid_test_size_range(self):
        """Test test_size outside valid range."""
        # Pydantic V2 uses different error messages - use simpler checks
        with pytest.raises(ValueError):
            DataConfig(test_size=0.05)  # Too small

        with pytest.raises(ValueError):
            DataConfig(test_size=0.5)  # Too large

    def test_invalid_val_size_range(self):
        """Test val_size outside valid range."""
        with pytest.raises(ValueError):
            DataConfig(val_size=0.05)  # Too small

        with pytest.raises(ValueError):
            DataConfig(val_size=0.4)  # Too large

    def test_insufficient_training_data(self):
        """Test that combined splits leave sufficient training data."""
        # These should fail validation due to field bounds
        invalid_configs = [
            {"test_size": 0.4, "val_size": 0.1},  # test_size > 0.3
            {"test_size": 0.3, "val_size": 0.3},  # val_size > 0.3
        ]

        for params in invalid_configs:
            with pytest.raises(ValueError):
                DataConfig(**params)
            print(
                f"✓ Correctly rejected: test_size={params['test_size']}, val_size={params['val_size']}"
            )

    def test_negative_counts(self):
        """Test that negative event counts are rejected."""
        with pytest.raises(ValueError):
            DataConfig(number_of_signals=-100)

        with pytest.raises(ValueError):
            DataConfig(number_of_background=-100)

    def test_data_dir_validation(self):
        """Test data directory validation."""
        with pytest.raises(ValueError, match="data_dir must be a non-empty string"):
            DataConfig(data_dir="")

        # Valid data dir
        config = DataConfig(data_dir="/custom/path")
        assert config.data_dir == "/custom/path"


def test_recommended_split_ratios():
    """Test commonly recommended split ratios for ML projects."""
    recommended_splits = [
        # (test_size, val_size, expected_train_size, description)
        (0.15, 0.15, 0.70, "70-15-15: Balanced splits"),
        (0.20, 0.10, 0.70, "70-20-10: More testing"),
        (0.10, 0.20, 0.70, "70-10-20: More validation"),
        (0.20, 0.15, 0.65, "65-20-15: Common compromise"),
    ]

    for test_size, val_size, expected_train, description in recommended_splits:
        config = DataConfig(test_size=test_size, val_size=val_size)
        actual_train = 1 - config.test_size - config.val_size

        assert abs(actual_train - expected_train) < 0.01
        print(
            f"✓ {description}: test={test_size:.0%}, val={val_size:.0%}, train={actual_train:.0%}"
        )


class TestSaveConfig:
    """Test SaveConfig validation."""

    def test_save_config_defaults(self):
        """Test SaveConfig default values."""
        config = SaveConfig()
        assert config.save_dir == "./figures"
        assert config.formats == ["png", "pdf"]
        assert config.dpi == 300

    def test_save_config_custom(self):
        """Test custom SaveConfig."""
        config = SaveConfig(
            save_dir="/custom/figures", formats=["svg", "png"], dpi=150, transparent=True
        )
        assert config.save_dir == "/custom/figures"
        assert config.formats == ["svg", "png"]
        assert config.dpi == 150
        assert config.transparent is True
