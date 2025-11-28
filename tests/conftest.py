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


@pytest.fixture
def sample_neutrino_data():
    """Fixture for neutrino data tests in test_config.py"""
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "signal": [1, 0, 1, 0, 1],
        }
    )
