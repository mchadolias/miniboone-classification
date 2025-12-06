import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.data_processor import DataProcessor
from src.config.config import DataConfig


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def config(tmp_path):
    """Minimal valid config for DataProcessor."""
    cfg = DataConfig(
        target_col="signal",
        cache_dir=tmp_path / "cache",
        scale_method="standard",
        variance_threshold=0.0,
        test_size=0.2,
        random_state=42,
        add_outlier_flag=True,  # Test outlier flag creation
        use_cache=False,
    )
    return cfg


@pytest.fixture
def processor(config):
    return DataProcessor(config)


# -----------------------------------------------------------------------------
# Test: Feature preparation
# -----------------------------------------------------------------------------
def test_prepare_features(processor, sample_neutrino_data):
    X, y = processor.prepare_features(sample_neutrino_data)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert processor.config.target_col not in X.columns
    assert len(X) == len(y)


# -----------------------------------------------------------------------------
# Test: Train/test split
# -----------------------------------------------------------------------------
def test_process_splits(processor, sample_neutrino_data):
    X_train, X_test, y_train, y_test = processor.process(sample_neutrino_data, to_numpy=False)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert len(X_train) + len(X_test) == len(sample_neutrino_data)


# -----------------------------------------------------------------------------
# Test: Pipeline creation occurs in process()
# -----------------------------------------------------------------------------
def test_pipeline_created(processor, sample_neutrino_data):
    assert processor.pipeline is None
    processor.process(sample_neutrino_data)
    assert processor.pipeline is not None


# -----------------------------------------------------------------------------
# Test: run_pipeline fit-transform behavior
# -----------------------------------------------------------------------------
def test_run_pipeline_transform(processor, sample_neutrino_data):
    X, _ = processor.prepare_features(sample_neutrino_data)
    X_transformed = processor.run_pipeline(X)

    # Should return pd.DataFrame or np.ndarray based on config
    assert isinstance(X_transformed, (pd.DataFrame, np.ndarray))

    # Shape must match (rows, n_features_after_transform)
    assert X_transformed.shape[0] == len(sample_neutrino_data)


# -----------------------------------------------------------------------------
# Test: Pipeline export + load
# -----------------------------------------------------------------------------
def test_export_and_load_pipeline(processor, sample_neutrino_data, tmp_path):
    processor.process(sample_neutrino_data)  # Fit pipeline

    path = processor.export_pipeline("test_pipeline.joblib")
    assert path.exists()

    # Load it in a new processor instance
    new_proc = DataProcessor(processor.config)
    new_proc.load_pipeline(path)

    assert new_proc.pipeline is not None


# -----------------------------------------------------------------------------
# Test: Outlier flag addition
# -----------------------------------------------------------------------------
def test_outlier_flag_added(processor, sample_neutrino_data_flag_added):
    X, _ = processor.prepare_features(sample_neutrino_data_flag_added)

    # Flag should be created only if config.add_outlier_flag = True
    assert "is_outlier" in X.columns


# -----------------------------------------------------------------------------
# Test: Test Process function type output
# -----------------------------------------------------------------------------
def test_process_returns_numpy(processor, sample_neutrino_data):
    """process(to_numpy=True) should return NumPy arrays for X splits."""
    X_train, X_test, y_train, y_test = processor.process(
        df=sample_neutrino_data,
        to_numpy=True,
    )

    assert isinstance(X_train, np.ndarray)
    assert isinstance(X_test, np.ndarray)

    # y should always remain a Series
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # Shapes should match
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)


def test_process_returns_dataframe(processor, sample_neutrino_data):
    """process(to_numpy=False) should return pandas DataFrames."""
    X_train, X_test, y_train, y_test = processor.process(
        df=sample_neutrino_data,
        to_numpy=False,
    )

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)

    # y always Series
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # Feature names must match the pipeline output
    assert list(X_train.columns) == processor.feature_names_ + (
        ["is_outlier"] if processor.config.add_outlier_flag else []
    )


# -----------------------------------------------------------------------------
# Test: Caching logic
# -----------------------------------------------------------------------------
@patch("joblib.dump")
def test_export_pipeline_calls_dump(mock_dump, processor, sample_neutrino_data, tmp_path):
    processor.process(sample_neutrino_data)
    path = processor.export_pipeline("pipeline.joblib")

    mock_dump.assert_called_once()
    assert "pipeline.joblib" in str(path)


# -----------------------------------------------------------------------------
# Test: Variance threshold removal
# -----------------------------------------------------------------------------
def test_variance_threshold_removal(tmp_path):
    """Feature with zero variance should be removed."""

    df = pd.DataFrame(
        {
            "feature_a": np.ones(100),  # zero variance
            "feature_b": np.random.randn(100),
            "signal": np.random.randint(0, 2, 100),
        }
    )

    cfg = DataConfig(
        target_col="signal",
        variance_threshold=0.1,
        cache_dir=tmp_path,
    )
    proc = DataProcessor(cfg)

    X_train, _, _, _ = proc.process(df, to_numpy=False)
    assert "feature_a" not in X_train.columns


# -----------------------------------------------------------------------------
# Test: Robust scaling option
# -----------------------------------------------------------------------------
def test_robust_scaling(tmp_path):
    """Verify that choosing 'robust' builds a RobustScaler."""
    df = pd.DataFrame(
        {
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100) * 5,
            "signal": np.random.randint(0, 2, 100),
        }
    )

    cfg = DataConfig(
        target_col="signal",
        scale_method="robust",
        cache_dir=tmp_path,
    )
    proc = DataProcessor(cfg)

    proc.process(df)
    assert "RobustScaler" in str(proc.pipeline)
