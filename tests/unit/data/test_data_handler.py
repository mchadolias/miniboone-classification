"""
Tests for the main MiniBooNEDataHandler class.

Covers initialization, data retrieval, preprocessing, cleaning,
and feature name extraction workflows with improved robustness.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.data.data_handler import MiniBooNEDataHandler


class TestMiniBooNEDataHandler:
    """Unit tests for the MiniBooNEDataHandler class."""

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------
    def test_initialization_defaults(self, empty_data_handler):
        """Handler initializes correctly with default attributes."""
        assert empty_data_handler.df is None
        assert isinstance(empty_data_handler.splits, dict)
        assert empty_data_handler.config.data_dir is not None
        assert hasattr(empty_data_handler, "downloader")
        assert hasattr(empty_data_handler, "loader")

    def test_config_integrity(self, empty_data_handler):
        """Ensure config attributes are valid."""
        cfg = empty_data_handler.config
        assert isinstance(cfg.data_dir, Path)
        assert isinstance(cfg.dataset, str)
        assert isinstance(cfg.test_size, float)
        assert isinstance(cfg.val_size, float)
        assert cfg.test_size + cfg.val_size < 0.5

    # -------------------------------------------------------------------------
    # Data Loading & Downloading
    # -------------------------------------------------------------------------
    def test_get_data_downloads_when_missing(self, empty_data_handler):
        """get_data() should trigger download when file missing."""
        with patch("os.path.exists", return_value=False):
            with (
                patch.object(empty_data_handler.downloader, "download") as mock_download,
                patch.object(empty_data_handler.loader, "load", return_value=pd.DataFrame()),
            ):
                empty_data_handler.get_data()
                mock_download.assert_called_once()

    def test_get_data_uses_existing_file(self, empty_data_handler):
        """get_data() uses local file when already present."""
        with patch("os.path.exists", return_value=True):
            with (
                patch.object(empty_data_handler.downloader, "download") as mock_download,
                patch.object(empty_data_handler.loader, "load", return_value=pd.DataFrame()),
            ):
                empty_data_handler.get_data()
                mock_download.assert_not_called()

    def test_get_data_returns_dataframe(self, empty_data_handler):
        """get_data() returns a valid DataFrame."""
        mock_df = pd.DataFrame({"feature_1": [1, 2, 3], "signal": [0, 1, 0]})
        with (
            patch("os.path.exists", return_value=True),
            patch.object(empty_data_handler.loader, "load", return_value=mock_df),
        ):
            df = empty_data_handler.get_data()
            assert isinstance(df, pd.DataFrame)
            assert "signal" in df.columns

    # -------------------------------------------------------------------------
    # Preprocessing & Splitting
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Feature Extraction
    # -------------------------------------------------------------------------
    def test_get_feature_names_excludes_target(self, data_handler_with_loaded_data):
        """Feature names should not include target variable."""
        features = data_handler_with_loaded_data.get_feature_names()
        assert "signal" not in features
        assert isinstance(features, tuple)
        assert len(features) > 0

    def test_get_feature_names_empty_dataframe(self, empty_data_handler):
        """Should raise if DataFrame is not loaded."""
        with pytest.raises(ValueError):
            empty_data_handler.get_feature_names()
