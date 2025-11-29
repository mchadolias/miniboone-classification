# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from src.data.data_handler import MiniBooNEDataHandler
from src.config import DataConfig


@pytest.fixture
def empty_data_handler():
    """Fresh DataHandler with no data loaded."""
    return MiniBooNEDataHandler()


@pytest.fixture
def data_handler_with_loaded_data():
    """DataHandler with sample data already loaded."""
    handler = MiniBooNEDataHandler()

    # Create sample data
    num_samples = 200
    data = {f"col_{i}": np.random.normal(0, 1, num_samples) for i in range(50)}
    handler.df = pd.DataFrame(data)

    # Add signal column (28% signal like real data)
    num_signals = int(num_samples * 0.28)
    handler.df["signal"] = (handler.df.index < num_signals).astype(int)

    # Update config
    handler.config.number_of_signals = num_signals
    handler.config.number_of_background = num_samples - num_signals

    return handler


@pytest.fixture
def sample_dataframe():
    """Generic sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "signal": [1, 0, 1, 0, 1],
        }
    )


@pytest.fixture
def small_dummy_data():
    """Small dataset for fast tests."""
    data = {f"col_{i}": [1, 2, 3, 4, 5] for i in range(50)}
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def sample_neutrino_data():
    """Sample neutrino dataset for plotting tests."""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, n_samples),
            "feature_2": np.random.exponential(2, n_samples),
            "feature_3": np.random.uniform(-5, 5, n_samples),
            "signal": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        }
    )

    # Add some outliers
    data["feature_1"][:5] = np.random.normal(10, 1, 5)  # Outliers
    data["feature_2"][:3] = np.random.exponential(10, 3)  # Outliers

    return data


@pytest.fixture
def plotter_config():
    """Configuration for plotter tests."""
    return {"figsize": (8, 6), "max_features": 5, "outlier_percentile": 5.0}
