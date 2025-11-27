import pytest
import pandas as pd
import numpy as np
from src.data.data_handler import MiniBooNEDataHandler
from src.config import DataConfig


@pytest.fixture
def empty_handler():
    """Fresh DataHandler with no data loaded."""
    return MiniBooNEDataHandler()


@pytest.fixture
def dummy_miniboone_data():
    """Raw dummy data without signal column (as it comes from CSV)."""
    num_samples = 200
    data = {f"col_{i}": np.random.normal(0, 1, num_samples) for i in range(50)}
    return pd.DataFrame(data)


@pytest.fixture
def dummy_miniboone_after_download():
    """Raw dummy data without signal column (as it comes from CSV)."""
    num_samples = 200
    data = {f"col_{i}": np.random.normal(0, 1, num_samples) for i in range(50)}
    return pd.DataFrame(data)


@pytest.fixture
def loaded_handler(dummy_miniboone_data):
    """DataHandler with data already loaded."""
    handler = MiniBooNEDataHandler()
    handler.df = dummy_miniboone_data
    return handler


@pytest.fixture
def handler_with_signal_data(dummy_miniboone_data):
    """DataHandler with signal column created (post-load state)."""
    handler = MiniBooNEDataHandler()
    handler.df = dummy_miniboone_data.copy()
    # Simulate what load() does
    handler.df.columns = [f"col_{i}" for i in range(50)]  # Rename columns

    # Create proper signal/background mix
    num_signals = int(len(handler.df) * 0.28)  # ~28% signal like real data
    handler.df["signal"] = (handler.df.index < num_signals).astype(int)

    # Update config to match
    handler.config.number_of_signals = num_signals
    handler.config.number_of_background = len(handler.df) - num_signals

    return handler


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
