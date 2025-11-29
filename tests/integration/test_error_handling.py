"""
Tests for error conditions and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_handler import MiniBooNEDataHandler
from src.visualization.plotter import NeutrinoPlotter


class TestErrorHandling:
    """Test error conditions and exception handling."""

    def test_plotter_invalid_data_types(self):
        """Test plotter with invalid data types."""
        plotter = NeutrinoPlotter()

        with pytest.raises(TypeError):
            plotter.create_target_distribution_plot("not_a_dataframe")

        with pytest.raises(TypeError):
            plotter.create_target_distribution_plot([1, 2, 3])

    def test_data_handler_invalid_config(self):
        """Test data handler with invalid configuration."""
        with pytest.raises(Exception):
            # This should fail during initialization
            handler = MiniBooNEDataHandler(config="invalid_config")

    def test_plotter_extreme_outlier_cases(self):
        """Test plotter with extreme outlier scenarios."""
        plotter = NeutrinoPlotter()

        # Data with only outliers
        extreme_data = pd.DataFrame(
            {"feature_1": [1000, 1001, 1002, 1003], "signal": [0, 1, 0, 1]}  # All extreme values
        )

        # Should handle gracefully
        fig = plotter.create_horizontal_boxplot_with_density(
            extreme_data, outlier_handling="remove", outlier_percentile=10.0
        )
        assert fig is not None

    def test_empty_after_outlier_removal(self):
        """Test case where outlier removal leaves no data."""
        plotter = NeutrinoPlotter()

        # Single extreme value
        data = pd.DataFrame({"feature_1": [1000], "signal": [1]})

        # Should handle empty data after outlier removal
        fig = plotter.create_horizontal_boxplot_with_density(
            data, outlier_handling="remove", outlier_percentile=1.0
        )
        assert fig is not None
