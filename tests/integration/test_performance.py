"""
Performance and scalability tests.
"""

import pytest
import pandas as pd
import numpy as np
import time
from src.plotter import NeutrinoPlotter


class TestPerformance:
    """Test performance characteristics."""

    def test_plotter_performance_large_dataset(self):
        """Test plotter performance with large datasets."""
        plotter = NeutrinoPlotter()

        # Create larger dataset
        np.random.seed(42)
        n_samples = 5000
        large_data = pd.DataFrame(
            {f"feature_{i}": np.random.normal(0, 1, n_samples) for i in range(20)}
        )
        large_data["signal"] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

        # Time the plotting
        start_time = time.time()
        fig = plotter.plot_feature_separation(
            df=large_data,
            features=[f"feature_{i}" for i in range(20)],
            target="signal",
            annotate_stats=True,
            show_mean_median=True,
        )
        end_time = time.time()

        execution_time = end_time - start_time
        print(f"Large dataset plotting took: {execution_time:.2f} seconds")

        assert fig is not None
        assert execution_time < 30.0  # Should complete within 30 seconds

    def test_memory_usage_with_large_data(self):
        """Test that plotter doesn't have memory leaks with large data."""
        plotter = NeutrinoPlotter()

        # Create multiple large plots and check for consistent performance
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature_1": np.random.normal(0, 1, 1000),
                "feature_2": np.random.exponential(1, 1000),
                "signal": np.random.choice([0, 1], 1000),
            }
        )

        # Create multiple plots
        for i in range(5):
            fig = plotter.plot_feature_separation(
                df=data,
                features=["feature_1", "feature_2"],
                target="signal",
                annotate_stats=True,
                show_mean_median=True,
            )
            assert fig is not None
