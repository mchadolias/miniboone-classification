"""
Tests for error conditions and edge cases.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.data_handler import MiniBooNEDataHandler
from src.plotter import NeutrinoPlotter


class TestErrorHandling:
    """Test error conditions and exception handling."""

    def test_plotter_invalid_data_types(self):
        """Test plotter with invalid data types."""
        plotter = NeutrinoPlotter()

        with pytest.raises(TypeError):
            plotter.plot_target_distribution("not_a_dataframe")

        with pytest.raises(TypeError):
            plotter.plot_target_distribution([1, 2, 3])

    def test_data_handler_invalid_config(self):
        """Test data handler with invalid configuration."""
        with pytest.raises(Exception):
            # This should fail during initialization
            handler = MiniBooNEDataHandler(config="invalid_config")
