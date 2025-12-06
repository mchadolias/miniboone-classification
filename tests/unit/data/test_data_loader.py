import os
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.data.data_loader import (
    DataLoader,
    LocalFileDownloader,
    KaggleDownloader,
    CUSTOM_COLUMN_NAMES,
)
from src.config.config import DataConfig


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def tmp_csv(tmp_path):
    """Create a dummy MiniBooNE-like CSV with 50 numeric columns."""
    df = pd.DataFrame({f"{i}": range(10) for i in range(50)})
    csv_path = tmp_path / "MiniBooNE_PID.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def config():
    """Minimal valid DataConfig."""
    return DataConfig(
        data_dir=Path("/tmp"),
        number_of_signals=5,
        number_of_background=5,
        target_col="signal",
    )


@pytest.fixture
def loader(config):
    return DataLoader(config)


# -----------------------------------------------------------------------------
# Test LocalFileDownloader
# -----------------------------------------------------------------------------
def test_local_file_downloader_success(tmp_csv):
    downloader = LocalFileDownloader(str(tmp_csv))
    path = downloader.download("/unused")
    assert path == str(tmp_csv)


def test_local_file_downloader_missing_file():
    downloader = LocalFileDownloader("nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        downloader.download("/unused")


# -----------------------------------------------------------------------------
# Test KaggleDownloader (fully mocked)
# -----------------------------------------------------------------------------
@patch("kaggle.api.kaggle_api_extended.KaggleApi")
def test_kaggle_downloader(mock_api, tmp_path):
    """
    Ensure KaggleDownloader:
        - Authenticates
        - Calls dataset_download_files
        - Extracts ZIP
        - Returns expected CSV
    """
    kaggle_dir = tmp_path / "data"
    kaggle_dir.mkdir()

    # Mock Kaggle API behavior -----------------------------------------------
    instance = mock_api.return_value
    instance.dataset_download_files.return_value = None

    # Create a fake zip file to simulate download
    zip_path = kaggle_dir / "dummy.zip"
    dummy_csv_path = kaggle_dir / "MiniBooNE_PID.csv"

    import zipfile

    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("MiniBooNE_PID.csv", "col1,col2\n1,2")

    downloader = KaggleDownloader("alexanderliapatis/miniboone")

    returned_path = downloader.download(str(kaggle_dir))

    assert Path(returned_path).exists()
    assert Path(returned_path).name == "MiniBooNE_PID.csv"

    instance.authenticate.assert_called_once()
    instance.dataset_download_files.assert_called_once()


# -----------------------------------------------------------------------------
# Test DataLoader.load()
# -----------------------------------------------------------------------------
def test_dataloader_load_success(tmp_csv, loader):
    df = loader.load(str(tmp_csv))

    # Column renaming
    assert list(df.columns[:-1]) == CUSTOM_COLUMN_NAMES

    # Target created
    assert loader.config.target_col in df.columns
    assert df[loader.config.target_col].sum() == loader.config.number_of_signals


def test_dataloader_raises_if_file_missing(loader):
    with pytest.raises(FileNotFoundError):
        loader.load("invalid.csv")


def test_dataloader_raises_if_column_count_mismatch(tmp_path, loader):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(bad_csv, index=False)

    with pytest.raises(ValueError):
        loader.load(str(bad_csv))


# -----------------------------------------------------------------------------
# Test validation inside DataLoader
# -----------------------------------------------------------------------------
def test_validate_loaded_data_logs(loader, tmp_csv, caplog):
    df = loader.load(str(tmp_csv))

    assert "Signal events" in caplog.text
    assert "Background events" in caplog.text
