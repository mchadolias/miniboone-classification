"""
Tests for the main MiniBooNEDataHandler facade class.
"""

import pytest
import pandas as pd
from src.data.data_handler import MiniBooNEDataHandler
from unittest.mock import patch, MagicMock


class TestMiniBooNEDataHandler:
    """Test suite for the main data handler facade."""

    def test_initialization(self, empty_data_handler):
        """Test that handler initializes with proper defaults."""
        assert empty_data_handler.df is None
        assert empty_data_handler.splits == {}
        assert empty_data_handler.config.data_dir is not None

    def test_get_data_downloads_when_missing(self, empty_data_handler):
        """Test get_data() downloads when file doesn't exist."""
        with patch("os.path.exists", return_value=False):
            with patch.object(empty_data_handler.downloader, "download") as mock_download:
                with patch.object(empty_data_handler.loader, "load") as mock_load:
                    mock_load.return_value = pd.DataFrame()
                    empty_data_handler.get_data()
                    mock_download.assert_called_once()

    def test_get_data_uses_existing_file(self, empty_data_handler):
        """Test get_data() uses existing file when available."""
        with patch("os.path.exists", return_value=True):
            with patch.object(empty_data_handler.downloader, "download") as mock_download:
                with patch.object(empty_data_handler.loader, "load") as mock_load:
                    mock_load.return_value = pd.DataFrame()
                    empty_data_handler.get_data()
                    mock_download.assert_not_called()

    def test_clean_data_requires_loaded_data(self, empty_data_handler):
        """Test clean_data() raises error when no data loaded."""
        with pytest.raises(ValueError, match="Data not loaded"):
            empty_data_handler.clean_data()

    def test_preprocess_creates_correct_splits(self, data_handler_with_loaded_data):
        """Test preprocess() creates train/val/test splits."""
        splits = data_handler_with_loaded_data.preprocess()
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"][0]) > 0  # Has samples

    def test_get_feature_names_excludes_target(self, data_handler_with_loaded_data):
        """Test get_feature_names() returns only feature columns."""
        features = data_handler_with_loaded_data.get_feature_names()
        assert "signal" not in features
        assert len(features) == 50
