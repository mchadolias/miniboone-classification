from abc import ABC, abstractmethod
import logging
import os
import zipfile

import pandas as pd

from src.config.config import DataConfig

logger = logging.getLogger(__name__)

# Default column names for the 50 MiniBooNE features
CUSTOM_COLUMN_NAMES = [f"feature_{i}" for i in range(50)]


class DataDownloader(ABC):
    """
    Abstract base class for data downloaders.

    This defines the contract that all data download strategies must implement,
    allowing flexible data sources (Kaggle, local files, URLs) while maintaining
    a consistent interface for the data pipeline.
    """

    @abstractmethod
    def download(self, data_dir: str) -> str:
        """
        Download data from the implemented source.

        Parameters
        ----------
        data_dir : str
            Directory where data should be downloaded.

        Returns
        -------
        str
            Path to the downloaded data file.

        """
        raise NotImplementedError


class KaggleDownloader(DataDownloader):
    """
    Downloader implementation for Kaggle datasets.

    Handles authentication with the Kaggle API and downloading of datasets
    from the Kaggle platform. This implementation is tailored to the
    MiniBooNE dataset structure.
    """

    def __init__(self, dataset: str):
        """
        Initialize Kaggle downloader with dataset identifier.

        Parameters
        ----------
        dataset : str
            Kaggle dataset identifier in 'username/dataset-name' format,
            e.g. 'alexanderliapatis/miniboone'.
        api : KaggleApi
            Instance of the Kaggle API client.
        """
        self.dataset = dataset
        self.api = {}

    def download(self, data_dir: str) -> str:
        """
        Download dataset from Kaggle and perform post-processing.

        This method:
        1. Authenticates with the Kaggle API
        2. Downloads and unzips the dataset
        3. Cleans up temporary zip files
        4. Returns path to the main data file (MiniBooNE_PID.csv)

        Parameters
        ----------
        data_dir : str
            Directory where data should be downloaded.

        Returns
        -------
        str
            Path to the downloaded MiniBooNE_PID.csv file.
        """
        try:
            logger.info("Authenticating with Kaggle API...")
            from kaggle.api.kaggle_api_extended import KaggleApi

            try:
                self.api = KaggleApi()
                self.api.authenticate()
            except Exception as auth_exc:
                logger.warning(
                    "Kaggle authentication failed. Ensure Kaggle API credentials are set up correctly. "
                    "Refer to https://www.kaggle.com/docs/api for guidance."
                )
                raise auth_exc

            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"Downloading dataset '{self.dataset}' to {data_dir}...")
            self.api.dataset_download_files(self.dataset, path=data_dir, unzip=False)

            # Extract and clean up zip files
            for fname in os.listdir(data_dir):
                if fname.endswith(".zip"):
                    zip_path = os.path.join(data_dir, fname)
                    logger.info(f"Extracting {zip_path}...")
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        zf.extractall(data_dir)
                    os.remove(zip_path)

            data_file = os.path.join(data_dir, "MiniBooNE_PID.csv")
            if not os.path.exists(data_file):
                raise FileNotFoundError(
                    f"Expected MiniBooNE_PID.csv in {data_dir}, but it was not found."
                )

            logger.info("Kaggle download completed successfully.")
            return data_file

        except Exception as e:
            logger.error(f"Failed to download dataset from Kaggle: {e}")
            raise RuntimeError(f"Failed to download dataset from Kaggle: {e}") from e


class LocalFileDownloader(DataDownloader):
    """
    Downloader for local files (primarily for testing or offline workflows).

    This implementation simply validates that a local file exists and
    returns its path, avoiding network calls during testing.
    """

    def __init__(self, file_path: str):
        """
        Initialize local file downloader.

        Parameters
        ----------
        file_path : str
            Path to an existing local data file.
        """
        self.file_path = file_path

    def download(self, data_dir: str) -> str:  # data_dir kept for interface compatibility
        """
        Validate local file exists and return its path.

        Parameters
        ----------
        data_dir : str
            Unused for local files (kept for interface consistency).

        Returns
        -------
        str
            Path to the local data file.

        """
        if not os.path.exists(self.file_path):
            logger.error(f"Local file not found: {self.file_path}")
            raise FileNotFoundError(f"Local file not found: {self.file_path}")

        logger.info(f"Using local dataset at {self.file_path}")
        return self.file_path


class DataLoader:
    """
    Handles data loading and basic validation for the MiniBooNE dataset.

    Responsibilities
    ----------------
    - Read the raw CSV file produced by MiniBooNE / Kaggle.
    - Validate that the number of columns matches expectations.
    - Apply standardized feature column names.
    - Create a binary target column (signal vs background) based on config.
    - Validate signal/background counts against configuration.

    This loader intentionally keeps the loading logic simple and physics-agnostic
    beyond the signal/background split; more advanced transformations are handled
    by DataCleaner and DataProcessor.
    """

    def __init__(self, config: DataConfig):
        """
        Initialize DataLoader with configuration.

        Parameters
        ----------
        config : DataConfig
            Data configuration containing feature counts, signal/background
            expectations, and target column name.
        """
        self.config = config
        self.logger = logger

    def load(self, data_file: str) -> pd.DataFrame:
        """
        Load and validate dataset from file.

        Steps
        -----
        1. Read CSV file into a DataFrame.
        2. Validate feature column count (expected 50 by default).
        3. Apply standardized column names (feature_0, feature_1, ..., feature_49).
        4. Create binary target column based on configured number_of_signals.
        5. Validate signal/background counts and log any mismatches.

        Parameters
        ----------
        data_file : str
            Path to the raw MiniBooNE data file.

        Returns
        -------
        pd.DataFrame
            Loaded and validated dataset with feature columns and target column.

        Raises
        ------
        FileNotFoundError
            If the data file does not exist.
        ValueError
            If the dataset structure does not match expectations.
        """
        if not os.path.exists(data_file):
            self.logger.error(f"Data file not found: {data_file}")
            raise FileNotFoundError(f"Data file not found: {data_file}")

        self.logger.info(f"Loading dataset: {data_file}...")
        df = pd.read_csv(data_file)

        # Determine expected number of feature columns (default 50)
        expected_features = getattr(self.config, "n_features", 50)

        # Validate dataset structure
        if len(df.columns) != expected_features:
            raise ValueError(f"Expected {expected_features} columns, got {len(df.columns)}")

        # Apply standardized feature column names
        if expected_features != len(CUSTOM_COLUMN_NAMES):
            # If config says something else, generate dynamically
            feature_names = [f"feature_{i}" for i in range(expected_features)]
        else:
            feature_names = CUSTOM_COLUMN_NAMES

        df.columns = feature_names

        # Target column name from config (default to 'signal' for backward compatibility)
        target_col = getattr(self.config, "target_col", "signal")

        # First N rows are signal events, remainder are background
        df[target_col] = (df.index < self.config.number_of_signals).astype(int)

        # Validate loaded data against config expectations
        self._validate_loaded_data(df, target_col=target_col)

        return df

    def _validate_loaded_data(self, df: pd.DataFrame, target_col: str = "signal") -> None:
        """
        Validate loaded dataset against expected configuration.

        Compares actual signal/background counts with configured expectations
        and logs warnings for mismatches while allowing processing to continue.

        Parameters
        ----------
        df : pd.DataFrame
            Loaded DataFrame to validate.
        target_col : str, default="signal"
            Name of the target column used for signal/background labeling.
        """
        signal_count = (df[target_col] == 1).sum()
        background_count = (df[target_col] == 0).sum()

        self.logger.info(
            f"Loaded {len(df):,} rows, {len(df.columns)} columns "
            f"({len(df.columns) - 1} features + 1 target)."
        )
        self.logger.info(f"Signal events: {signal_count:,}")
        self.logger.info(f"Background events: {background_count:,}")

        # Provide warnings for configuration mismatches
        if signal_count != self.config.number_of_signals:
            self.logger.warning(
                "Signal count mismatch. Expected %s, got %s",
                f"{self.config.number_of_signals:,}",
                f"{signal_count:,}",
            )

        if background_count != self.config.number_of_background:
            self.logger.warning(
                "Background count mismatch. Expected %s, got %s",
                f"{self.config.number_of_background:,}",
                f"{background_count:,}",
            )
