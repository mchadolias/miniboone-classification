# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from src.data.data_handler import MiniBooNEDataHandler


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
def synthetic_miniboone_data():
    """Create a synthetic MiniBooNE-like dataset for testing."""
    n_samples = 200
    n_features = 8

    # Create synthetic data with similar characteristics to MiniBooNE
    np.random.seed(42)  # For reproducible tests

    # Feature 1-2: Normally distributed (like some physical measurements)
    feature_1 = np.random.normal(0, 1, n_samples)
    feature_2 = np.random.normal(5, 2, n_samples)

    # Feature 3-4: Highly skewed (common in particle physics)
    feature_3 = np.random.exponential(2, n_samples)
    feature_4 = np.random.gamma(2, 2, n_samples)

    # Feature 5: Zero-inflated (common in detector readings)
    feature_5 = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    feature_5 = feature_5 * np.random.exponential(1, n_samples)

    # Feature 6-7: Correlated features
    feature_6 = np.random.normal(0, 1, n_samples)
    feature_7 = feature_6 + np.random.normal(0, 0.1, n_samples)

    # Feature 8: Mixed distribution
    feature_8 = np.concatenate(
        [np.random.normal(-2, 1, n_samples // 2), np.random.normal(2, 1, n_samples // 2)]
    )

    # Combine features
    features = np.column_stack(
        [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]
    )

    # Create target with ~30% signal (mimicking MiniBooNE ratio)
    signal_ratio = 0.3
    n_signal = int(n_samples * signal_ratio)
    signal = np.concatenate([np.ones(n_signal), np.zeros(n_samples - n_signal)])
    np.random.shuffle(signal)

    # Create DataFrame with meaningful column names
    feature_names = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(features, columns=feature_names)
    df["signal"] = signal.astype(int)

    # Add a few missing values for testing
    df.iloc[5, 2] = np.nan  # One missing value
    df.iloc[10, 4] = np.nan  # Another missing value

    return df


@pytest.fixture
def plotter_config():
    """Configuration for plotter tests."""
    return {"figsize": (8, 6), "max_features": 5, "outlier_percentile": 5.0}
