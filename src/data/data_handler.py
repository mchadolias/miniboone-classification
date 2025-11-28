# src/data/data_handler.py
"""
Data handling module for MiniBooNE particle classification dataset.

This module provides a clean architecture implementation for handling the complete
data pipeline from download to preprocessing. It uses the Strategy Pattern for
different data sources and the Facade Pattern to provide a simple interface.

Key Components:
- DataDownloader: Abstract interface for data sources
- DataLoader: Handles data loading and validation
- DataCleaner: Handles data quality checks
- DataPreprocessor: Handles feature scaling and splitting
- DataSaver: Handles saving processed data
- MiniBooNEDataHandler: Main facade that orchestrates the pipeline
"""

import os
from typing import Dict, Optional, Tuple, List
from abc import ABC, abstractmethod
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import DataConfig

# Constants
CUSTOM_COLUMN_NAMES = [f"col_{i}" for i in range(50)]


class DataDownloader(ABC):
    """
    Abstract base class for data downloaders.

    This defines the contract that all data download strategies must implement,
    allowing flexible data sources (Kaggle, local files, URLs) while maintaining
    a consistent interface for the data pipeline.

    Methods:
        download: Download data and return path to downloaded file
    """

    @abstractmethod
    def download(self, data_dir: str) -> str:
        """
        Download data from the implemented source.

        Args:
            data_dir: Directory where data should be downloaded

        Returns:
            str: Path to the downloaded data file

        Raises:
            RuntimeError: If download fails
            FileNotFoundError: If local file doesn't exist
        """
        pass


class KaggleDownloader(DataDownloader):
    """
    Downloader implementation for Kaggle datasets.

    Handles authentication with Kaggle API and downloading of datasets
    from the Kaggle platform. This implementation is specific to the
    MiniBooNE dataset structure.

    Attributes:
        dataset (str): Kaggle dataset identifier in 'username/dataset-name' format
    """

    def __init__(self, dataset: str):
        """
        Initialize Kaggle downloader with dataset identifier.

        Args:
            dataset: Kaggle dataset identifier (e.g., 'alexanderliapatis/miniboone')
        """
        self.dataset = dataset

    def download(self, data_dir: str) -> str:
        """
        Download dataset from Kaggle and perform post-processing.

        This method:
        1. Authenticates with Kaggle API
        2. Downloads and unzips the dataset
        3. Cleans up temporary zip files
        4. Returns path to the main data file

        Args:
            data_dir: Directory where data should be downloaded

        Returns:
            str: Path to the downloaded MiniBooNE_PID.csv file

        Raises:
            RuntimeError: If Kaggle API authentication or download fails
            ImportError: If kaggle package is not installed
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(self.dataset, path=data_dir, unzip=True)

            # Clean up zip files after extraction
            for f in os.listdir(data_dir):
                if f.endswith(".zip"):
                    with zipfile.ZipFile(os.path.join(data_dir, f), "r") as zf:
                        zf.extractall(data_dir)
                    os.remove(os.path.join(data_dir, f))

            data_file = os.path.join(data_dir, "MiniBooNE_PID.csv")
            print("✅ Download complete.")
            return data_file

        except Exception as e:
            raise RuntimeError(f"Failed to download dataset from Kaggle: {e}")


class LocalFileDownloader(DataDownloader):
    """
    Downloader for local files (primarily for testing).

    This implementation simply validates that a local file exists and
    returns its path, avoiding network calls during testing.

    Attributes:
        file_path (str): Path to local data file
    """

    def __init__(self, file_path: str):
        """
        Initialize local file downloader.

        Args:
            file_path: Path to existing local data file
        """
        self.file_path = file_path

    def download(self, data_dir: str) -> str:
        """
        Validate local file exists and return its path.

        Args:
            data_dir: Unused for local files (maintained for interface consistency)

        Returns:
            str: Path to the local data file

        Raises:
            FileNotFoundError: If local file doesn't exist
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Local file not found: {self.file_path}")
        print("✅ Using local file.")
        return self.file_path


class DataLoader:
    """
    Handles data loading and basic validation.

    Responsible for reading data files, applying column names, creating
    the target signal column, and validating dataset structure.

    Attributes:
        config (DataConfig): Configuration object with dataset parameters
    """

    def __init__(self, config: DataConfig):
        """
        Initialize data loader with configuration.

        Args:
            config: Data configuration containing signal counts and other parameters
        """
        self.config = config

    def load(self, data_file: str) -> pd.DataFrame:
        """
        Load and validate dataset from file.

        This method:
        1. Reads CSV file into DataFrame
        2. Validates column count matches expected features
        3. Applies standardized column names
        4. Creates binary signal column based on config
        5. Performs dataset validation

        Args:
            data_file: Path to data file

        Returns:
            pd.DataFrame: Loaded and validated dataset

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If dataset structure doesn't match expectations
        """
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        print("📂 Loading data...")
        df = pd.read_csv(data_file)

        # Validate dataset structure
        if len(df.columns) != 50:
            raise ValueError(f"Expected 50 columns, got {len(df.columns)}")

        # Apply standardized column names
        df.columns = CUSTOM_COLUMN_NAMES

        # Create binary signal column (first N rows are signal events)
        df["signal"] = (df.index < self.config.number_of_signals).astype(int)

        self._validate_loaded_data(df)
        return df

    def _validate_loaded_data(self, df: pd.DataFrame) -> None:
        """
        Validate loaded dataset against expected configuration.

        Compares actual signal/background counts with configured expectations
        and provides warnings for mismatches while allowing processing to continue.

        Args:
            df: Loaded DataFrame to validate
        """
        signal_count = (df["signal"] == 1).sum()
        background_count = (df["signal"] == 0).sum()

        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns.")
        print(f"📊 Signal events: {signal_count:,}")
        print(f"📊 Background events: {background_count:,}")

        # Provide warnings for configuration mismatches
        if signal_count != self.config.number_of_signals:
            print(
                f"⚠️  Warning: Signal count mismatch. Expected {self.config.number_of_signals:,}, got {signal_count:,}"
            )

        if background_count != self.config.number_of_background:
            print(
                f"⚠️  Warning: Background count mismatch. Expected {self.config.number_of_background:,}, got {background_count:,}"
            )


class DataCleaner:
    """
    Handles data cleaning operations.

    Performs basic data quality checks including missing value detection
    and duplicate removal. The MiniBooNE dataset is typically clean, but
    these checks provide robustness for data quality issues.
    """

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean dataset by handling missing values and duplicates.

        Args:
            df: DataFrame to clean

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count != 0:
            print(f"There are in total: {missing_count} missing values")
            # In a real scenario, you might impute or remove here
        else:
            print("✅ There are no missing values")

        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count != 0:
            print(f"There are in total {duplicate_count} duplicated values")
            df = df.drop_duplicates()
            print("✅ Dropped the duplicated values")
        else:
            print("✅ There are no duplicated values")

        return df


class DataPreprocessor:
    """
    Handles data preprocessing and splitting.

    Responsible for feature scaling and dataset splitting into train/validation/test
    sets. Uses stratified splitting to maintain class distribution and standard
    scaling for feature normalization.

    Attributes:
        config (DataConfig): Configuration with split sizes and random state
        scaler (StandardScaler): Fitted scaler for feature standardization
        splits (Dict): Processed data splits
    """

    def __init__(self, config: DataConfig):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Data configuration containing test/val sizes and random state
        """
        self.config = config
        self.scaler = StandardScaler()
        self.splits: Dict[str, Tuple[np.ndarray, pd.Series]] = {}

    def preprocess(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, pd.Series]]:
        """
        Preprocess and split data into train/validation/test sets.

        This method:
        1. Splits data into train+val and test sets (stratified)
        2. Further splits train+val into train and validation
        3. Applies standard scaling to features
        4. Returns processed splits

        Important: Uses stratified splitting to maintain signal/background
        ratio across all splits.

        Args:
            df: DataFrame to preprocess

        Returns:
            Dictionary with 'train', 'val', 'test' splits as tuples of
            (scaled_features, target_series)
        """

        print("🔄 Preprocessing data...")

        X = df.drop("signal", axis=1)
        y = df["signal"]

        # Split into train+val and test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        # Split train+val into train and val
        val_rel_size = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_rel_size,
            random_state=self.config.random_state,
            stratify=y_trainval,
        )

        # Standardize features
        print("📏 Standardizing features...")
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # Store processed splits (only scaled data)
        self.splits = {
            "train": (X_train_scaled, y_train.reset_index(drop=True)),
            "val": (X_val_scaled, y_val.reset_index(drop=True)),
            "test": (X_test_scaled, y_test.reset_index(drop=True)),
        }

        # Store raw splits separately if needed
        self.raw_splits = {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }

        self._print_split_stats()
        print("✅ Preprocessing complete.")
        return self.splits

    def _print_split_stats(self) -> None:
        """
        Print statistics about the processed data splits.
        """
        if not self.splits:
            return

        print("📊 Data Split Statistics:")
        for split_name, (X, y) in self.splits.items():
            signal_pct = (y == 1).mean() * 100
            print(f"   {split_name.capitalize()}: {len(X):,} samples ({signal_pct:.1f}% signal)")


class DataSaver:
    """
    Handles saving processed data to disk.

    Saves processed splits, fitted scaler, and feature names for
    reproducible model training and inference.
    """

    def save_processed_data(
        self,
        splits: Dict[str, Tuple[np.ndarray, pd.Series]],
        scaler: StandardScaler,
        feature_names: List[str],
        output_dir: str = "processed_data",
    ) -> None:
        """
        Save processed splits and artifacts to disk.

        Args:
            splits: Processed data splits from preprocessor
            scaler: Fitted StandardScaler for feature transformation
            feature_names: List of feature column names
            output_dir: Directory to save processed data

        Raises:
            ValueError: If no processed data is available
        """
        if not splits:
            raise ValueError("No processed data available.")

        os.makedirs(output_dir, exist_ok=True)

        # Save processed splits as numpy arrays
        for split_name, (X, y) in splits.items():
            if split_name != "raw":  # Don't save raw data
                np.save(os.path.join(output_dir, f"X_{split_name}.npy"), X)
                np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y.values)

        # Save preprocessing artifacts
        import joblib

        joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(feature_names, os.path.join(output_dir, "feature_names.pkl"))

        print(f"💾 Processed data saved to {output_dir}/")


class MiniBooNEDataHandler:
    """
    Main facade class for MiniBooNE dataset operations.

    Orchestrates the complete data pipeline from download to preprocessing
    using the Strategy Pattern for different components. Provides a simple
    interface while allowing flexible implementation swapping for testing
    and different environments.

    The class follows the Facade Pattern to hide complex subsystem interactions
    and provide a clean, high-level API for data operations.

    Example:
        >>> handler = MiniBooNEDataHandler()
        >>> df = handler.get_data()           # Download and load data
        >>> handler.clean_data()              # Clean the dataset
        >>> splits = handler.preprocess()     # Preprocess and split
        >>> handler.save_processed_data()     # Save processed data

    Attributes:
        config (DataConfig): Configuration object for data parameters
        downloader (DataDownloader): Strategy for data downloading
        loader (DataLoader): Handles data loading and validation
        cleaner (DataCleaner): Handles data cleaning operations
        preprocessor (DataPreprocessor): Handles preprocessing and splitting
        saver (DataSaver): Handles saving processed data
        df (pd.DataFrame): Loaded dataset (None until data is loaded)
        splits (Dict): Preprocessed data splits (empty until preprocessing)
    """

    def __init__(
        self, config: Optional[DataConfig] = None, downloader: Optional[DataDownloader] = None
    ):
        """
        Initialize the data handler with optional dependency injection.

        This constructor demonstrates Dependency Injection principle, allowing
        different downloader strategies to be injected for testing or different
        data sources.

        Args:
            config: Data configuration object. If None, uses default DataConfig.
            downloader: Data downloader strategy. If None, uses KaggleDownloader.
        """
        self.config = config or DataConfig()

        # Use dependency injection for downloader with fallback
        self.downloader = downloader or KaggleDownloader(
            getattr(self.config, "dataset", "alexanderliapatis/miniboone")
        )

        # Initialize component strategies
        self.loader = DataLoader(self.config)
        self.cleaner = DataCleaner()
        self.preprocessor = DataPreprocessor(self.config)
        self.saver = DataSaver()

        # Initialize state
        self.df: Optional[pd.DataFrame] = None
        self.splits: Dict[str, Tuple[np.ndarray, pd.Series]] = {}

        # Ensure data directory exists
        os.makedirs(self.config.data_dir, exist_ok=True)

    def get_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Main entry point to get data (downloads if necessary).

        This method orchestrates the complete data acquisition pipeline:
        checks for existing data, downloads if missing or forced, loads,
        and returns the dataset.

        Args:
            force_download: If True, always download even if file exists

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            RuntimeError: If download fails
            FileNotFoundError: If data file doesn't exist and download fails
        """
        data_file = os.path.join(self.config.data_dir, "MiniBooNE_PID.csv")

        if force_download or not os.path.exists(data_file):
            print("⬇️  Downloading dataset...")
            data_file = self.downloader.download(self.config.data_dir)

        if self.df is None:
            self.df = self.loader.load(data_file)

        return self.df

    def download(self) -> None:
        """
        Convenience method to download data only.

        Useful for pre-downloading data without loading it into memory.
        """
        data_file = os.path.join(self.config.data_dir, "MiniBooNE_PID.csv")
        self.downloader.download(self.config.data_dir)

    def load(self) -> pd.DataFrame:
        """
        Convenience method to load data only.

        Assumes data file already exists locally from previous download.

        Returns:
            pd.DataFrame: Loaded dataset

        Raises:
            FileNotFoundError: If data file doesn't exist locally
        """
        data_file = os.path.join(self.config.data_dir, "MiniBooNE_PID.csv")
        self.df = self.loader.load(data_file)
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """
        Clean the loaded data for quality issues.

        Returns:
            pd.DataFrame: Cleaned dataset

        Raises:
            ValueError: If no data is loaded
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call get_data() or load() first.")

        self.df = self.cleaner.clean(self.df)
        return self.df

    def preprocess(self) -> Dict[str, Tuple[np.ndarray, pd.Series]]:
        """
        Preprocess loaded data into train/validation/test splits.

        Returns:
            Dictionary with processed splits ready for model training

        Raises:
            ValueError: If no data is loaded
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call get_data() or load() first.")

        self.splits = self.preprocessor.preprocess(self.df)
        return self.splits

    def save_processed_data(self, output_dir: str = "processed_data") -> None:
        """
        Save processed splits and preprocessing artifacts to disk.

        Args:
            output_dir: Directory to save processed data

        Raises:
            ValueError: If no processed data is available
        """
        if not self.splits:
            raise ValueError("No processed data available. Call preprocess() first.")

        self.saver.save_processed_data(
            splits=self.splits,
            scaler=self.preprocessor.scaler,
            feature_names=self.get_feature_names(),
            output_dir=output_dir,
        )

    def get_feature_names(self) -> List[str]:
        """
        Get the names of the feature columns (excluding target).

        Returns:
            List of feature column names

        Raises:
            ValueError: If no data is loaded
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call get_data() or load() first.")
        return [col for col in self.df.columns if col != "signal"]

    def get_data_summary(self) -> Dict[str, any]:
        """
        Get comprehensive summary of the loaded dataset.

        Returns:
            Dictionary with dataset statistics including sample counts,
            signal/background distribution, and feature information.

        Raises:
            ValueError: If no data is loaded
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call get_data() or load() first.")

        return {
            "total_samples": len(self.df),
            "signal_count": (self.df["signal"] == 1).sum(),
            "background_count": (self.df["signal"] == 0).sum(),
            "signal_ratio": (self.df["signal"] == 1).mean(),
            "feature_count": len(self.df.columns) - 1,  # exclude target
            "feature_names": self.get_feature_names(),
            "missing_values": self.df.isnull().sum().sum(),
        }
