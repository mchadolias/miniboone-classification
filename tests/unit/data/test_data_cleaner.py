import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data import DataCleaner


class TestDataCleaner:
    # -------------------------------------------------------------------------
    # Cleaning & Validation
    # -------------------------------------------------------------------------
    def test_clean_data_requires_loaded_data(self, empty_data_handler):
        """clean_data() should raise if data not loaded."""
        with pytest.raises(ValueError, match="Data not loaded"):
            empty_data_handler.clean_data()

    def test_clean_data_handles_nan_rows(self, data_handler_with_loaded_data):
        handler = data_handler_with_loaded_data
        handler.df.loc[0, "feature_1"] = np.nan
        handler.clean_data()
        assert not handler.df["feature_1"].isna().any()

    def test_duplicated_events_returns_dataframe(self, synthetic_miniboone_data):
        cleaner = DataCleaner()
        df_loaded = synthetic_miniboone_data.copy()

        # This should return a DataFrame with cleaned data
        df_cleaned = cleaner._handle_duplicated_events(df=df_loaded)

        # Verify it returns a proper DataFrame
        assert isinstance(df_cleaned, pd.DataFrame)
        assert df_cleaned is not None
        assert len(df_cleaned) <= len(df_loaded)  # Shouldn't have more rows

        # If no duplicates were present, they should be equal
        if df_loaded.duplicated().sum() == 0:
            pd.testing.assert_frame_equal(df_loaded, df_cleaned)
