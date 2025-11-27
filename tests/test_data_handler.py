import pytest
import pandas as pd
from src.data.data_handler import MiniBooNEDataHandler
from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi
from unittest.mock import patch, MagicMock


class TestDataHandler:
    def test_initialization(self, empty_handler):
        """Test that the class initializes with proper defaults"""
        assert empty_handler.df is None
        assert empty_handler.splits == {}
        assert empty_handler.data_dir is not None
        assert isinstance(empty_handler.scaler, StandardScaler)

    def test_get_feature_names(self, handler_with_signal_data):
        """Test feature names extraction"""
        features = handler_with_signal_data.get_feature_names()
        assert "signal" not in features
        assert len(features) == 50  # All 50 feature columns

    def test_validate_dataset(self, handler_with_signal_data):
        """Test dataset validation logic"""
        handler_with_signal_data.config.number_of_signals = 56  # 28% of 200
        handler_with_signal_data.config.number_of_background = 144

        # This should not raise an exception
        handler_with_signal_data._validate_dataset()

    def test_kaggle_api_authentication(self):
        """Test that Kaggle API can authenticate successfully."""
        # Skip in CI - no credentials available
        import os

        if os.getenv("CI"):
            pytest.skip("No Kaggle credentials in CI environment")

        try:
            api = KaggleApi()
            api.authenticate()
            assert True
        except Exception as e:
            pytest.fail(f"Kaggle API authentication failed: {e}")

    def test_kaggle_api_in_download_method(self, empty_handler):
        """Test that download method properly uses Kaggle API."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False  # File doesn't exist

            # Patch the KaggleApi in YOUR MODULE's namespace
            with patch("src.data.data_handler.KaggleApi") as mock_api_class:
                mock_api = MagicMock()
                mock_api_class.return_value = mock_api

                # Call the download method
                empty_handler.download()

                # Verify the API was called correctly
                mock_api.authenticate.assert_called_once()
                mock_api.dataset_download_files.assert_called_once_with(
                    empty_handler.config.dataset, path=empty_handler.data_dir, unzip=True
                )

    def test_data_load(self, empty_handler, dummy_miniboone_after_download):
        """Test that the file is loaded correctly and signal column is created properly."""
        # Set expected signal count in config
        expected_signals = 56  # 28% of 200 samples
        empty_handler.config.number_of_signals = expected_signals

        with patch("pandas.read_csv", return_value=dummy_miniboone_after_download):
            with patch("os.path.exists", return_value=True):
                loaded_df = empty_handler.load()

        # Test that data was loaded
        assert empty_handler.df is not None
        assert len(empty_handler.df) == len(dummy_miniboone_after_download)

        # Test that signal column was CREATED (not loaded from file)
        assert "signal" in empty_handler.df.columns
        assert empty_handler.df.shape[1] == 51  # 50 features + newly created signal column

        # Test signal column creation logic: first N rows are signal
        actual_signals = (empty_handler.df["signal"] == 1).sum()
        assert actual_signals == expected_signals
        assert actual_signals == empty_handler.config.number_of_signals

        # Verify the signal creation logic: first N indices should be signal=1
        assert all(
            empty_handler.df.iloc[:expected_signals]["signal"] == 1
        ), "First N rows should be signal"
        assert all(
            empty_handler.df.iloc[expected_signals:]["signal"] == 0
        ), "Remaining rows should be background"

        # Test column renaming
        expected_columns = [f"col_{i}" for i in range(50)] + ["signal"]
        assert list(empty_handler.df.columns) == expected_columns

        print(f"✅ Correctly created {actual_signals} signals from first {expected_signals} rows")

    def test_data_load_with_different_signal_counts(self, empty_handler):
        """Test that signal creation works with different config values."""
        test_cases = [
            (10, 40),  # 10 signals, 40 background
            (25, 75),  # 25 signals, 75 background
            (0, 100),  # All background
            (100, 0),  # All signal (edge case)
        ]

        for num_signals, num_background in test_cases:
            # Create a fresh handler for each test case to avoid state pollution
            handler = MiniBooNEDataHandler()
            handler.config.number_of_signals = num_signals

            total_samples = num_signals + num_background
            dummy_data = pd.DataFrame({f"col_{i}": range(total_samples) for i in range(50)})

            with patch("pandas.read_csv", return_value=dummy_data):
                with patch("os.path.exists", return_value=True):
                    handler.load()

            actual_signals = (handler.df["signal"] == 1).sum()
            actual_background = (handler.df["signal"] == 0).sum()

            assert actual_signals == num_signals
            assert actual_background == num_background
            assert len(handler.df) == total_samples

    def test_signal_column_creation(self, empty_handler):
        """Test the specific logic: signal = (index < number_of_signals)."""
        empty_handler.config.number_of_signals = 3  # Small number for easy testing

        # Create dummy data with 5 rows
        dummy_data = pd.DataFrame({f"col_{i}": [1, 2, 3, 4, 5] for i in range(50)})

        with patch("pandas.read_csv", return_value=dummy_data):
            with patch("os.path.exists", return_value=True):
                empty_handler.load()

        # Test the specific logic: first 3 rows (index 0,1,2) should be signal=1
        expected_signals = [1, 1, 1, 0, 0]  # First 3 are signal, rest background
        actual_signals = empty_handler.df["signal"].tolist()

        assert (
            actual_signals == expected_signals
        ), f"Expected {expected_signals}, got {actual_signals}"

    def test_methods_raise_error_when_data_not_loaded(self, empty_handler):
        """Test that methods raise ValueError when data is not loaded."""
        # Test clean_data()
        with pytest.raises(ValueError) as exc_info:
            empty_handler.clean_data()
        assert "Data not loaded" in str(exc_info.value)
        assert "Call load() first" in str(exc_info.value)

        # Test preprocess()
        with pytest.raises(ValueError) as exc_info:
            empty_handler.preprocess()
        assert "Data not loaded" in str(exc_info.value)
        assert "Call load() first" in str(exc_info.value)

        # Test get_feature_names()
        with pytest.raises(ValueError) as exc_info:
            empty_handler.get_feature_names()
        assert "Data not loaded" in str(exc_info.value)
        assert "Call load() first" in str(exc_info.value)

        # Test get_data_summary()
        with pytest.raises(ValueError) as exc_info:
            empty_handler.get_data_summary()
        assert "Data not loaded" in str(exc_info.value)
        assert "Call load() first" in str(exc_info.value)

    def test_clean_data_with_loaded_data(self, loaded_handler):
        """Test clean_data works when data is loaded."""
        result = loaded_handler.clean_data()
        assert result is not None
        assert loaded_handler.df is not None

    def test_get_data_summary(self, handler_with_signal_data):
        """Test data summary returns correct information."""
        summary = handler_with_signal_data.get_data_summary()

        assert summary["total_samples"] == len(handler_with_signal_data.df)
        assert summary["signal_count"] == (handler_with_signal_data.df["signal"] == 1).sum()
        assert summary["background_count"] == (handler_with_signal_data.df["signal"] == 0).sum()
        assert summary["feature_count"] == 50
        assert len(summary["feature_names"]) == 50

    def test_save_processed_data_requires_preprocess(self, handler_with_signal_data):
        """Test that save_processed_data requires preprocessed data."""
        with pytest.raises(ValueError, match="No processed data available"):
            handler_with_signal_data.save_processed_data()
