"""
MiniBooNEDataHandler
====================

Central orchestration layer for the MiniBooNE dataset lifecycle.

This class integrates and manages all core data operations:
downloading, loading, cleaning, and preprocessing. It ensures that each
step in the data pipeline is tracked, reproducible, and modular.

Pipeline Overview
-----------------

    :DataLoader:  →  :DataCleaner: → :DataProcessor:

Steps:
    1. Ensures dataset availability and loads dataset into pandas DataFrame
    2. Removes invalid entries, NaNs (-999), and duplicates
    3. Scales features, splits into train/test sets, and readies for modeling


Responsibilities
----------------
Implements the Facade Pattern to unify:
- Data download (via DataDownloader)
- Data loading (via DataLoader)
- Cleaning (via DataCleaner)
- Preprocessing (via DataProcessor)

Also provides convenience utilities for summarizing and exporting
processed data, and tracks scientific data lineage for reproducibility.

Author: M. Chadolias
Project: MiniBooNE Neutrino Classification
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.config.config import DataConfig
from src.data.data_cleaner import DataCleaner
from src.data.data_loader import DataDownloader, DataLoader, KaggleDownloader
from src.data.data_processor import DataProcessor


class MiniBooNEDataHandler:
    """
    Main facade class for MiniBooNE dataset operations.

    Orchestrates the complete data pipeline from download to preprocessing,
    using strategy classes for each subsystem:

        • DataDownloader  → obtains the raw dataset (Kaggle/local/etc.)
        • DataLoader      → reads and validates the raw file into a DataFrame
        • DataCleaner     → handles duplicates, NaNs, and -999 placeholders
        • DataProcessor   → scales features, splits data, and exports pipelines

    This class hides the internal complexity and offers a small, coherent API
    for experiments, notebooks, and training scripts.

    Parameters
    ----------
    config : DataConfig
        Configuration object controlling data paths, target column name,
        expected counts, cache directories, etc.
    downloader : Optional[DataDownloader], default=None
        Optional custom downloader strategy. If None, a KaggleDownloader
        will be constructed from config (if possible).
    loader : Optional[DataLoader], default=None
        Optional custom loader. If None, a default DataLoader(config) is used.
    cleaner : Optional[DataCleaner], default=None
        Optional custom cleaner. If None, a default DataCleaner(config.target_col) is used.
    processor : Optional[DataProcessor], default=None
        Optional custom processor. If None, a default DataProcessor(config) is used.

    Attributes
    ----------
    df : Optional[pd.DataFrame]
        Loaded (and optionally cleaned) dataset.
    splits : Dict[str, Tuple[np.ndarray, pd.Series]]
        Train/test splits as NumPy arrays (model-ready).
    splits_df : Dict[str, Tuple[pd.DataFrame, pd.Series]]
        Train/test splits as DataFrames (analysis/plot-ready).
    lineage : Dict[str, str]
        Dictionary of lineage metadata for scientific reproducibility.
    """

    def __init__(
        self,
        config: Optional[DataConfig] = None,
        downloader: Optional[DataDownloader] = None,
        loader: Optional[DataLoader] = None,
        cleaner: Optional[DataCleaner] = None,
        processor: Optional[DataProcessor] = None,
    ):
        self.config: DataConfig = config or DataConfig()

        # Strategy components
        self.downloader: DataDownloader = downloader or KaggleDownloader(
            getattr(self.config, "dataset", "alexanderliapatis/miniboone")
        )
        self.loader: DataLoader = loader or DataLoader(self.config)
        self.cleaner: DataCleaner = cleaner or DataCleaner(self.config.target_col)
        self.processor: DataProcessor = processor or DataProcessor(self.config)

        # Internal state
        self.df: Optional[pd.DataFrame] = None
        # NumPy-based splits (for model training)
        self.splits: Dict[str, Tuple[np.ndarray, pd.Series]] = {}
        # DataFrame-based splits (for inspection/plotting)
        self.splits_df: Dict[str, Tuple[pd.DataFrame, pd.Series]] = {}
        # Scientific lineage / provenance metadata
        self.lineage: Dict[str, Any] = {}

        # Setup logging
        self.logger = logging.getLogger(__name__)

        # Ensure base directories exist
        os.makedirs(getattr(self.config, "data_dir", "data"), exist_ok=True)
        os.makedirs(getattr(self.config, "cache_dir", "data/processed"), exist_ok=True)

        self.logger.info("MiniBooNEDataHandler initialized.")

    # -------------------------------------------------------------------------
    # DATA DOWNLOAD / LOAD
    # -------------------------------------------------------------------------
    def download(self) -> str:
        """
        Download the MiniBooNE dataset to the configured data directory.

        Returns
        -------
        str
            Path to the downloaded data file.
        """
        data_dir = getattr(self.config, "data_dir", "data")
        path = self.downloader.download(data_dir)
        self.lineage["download_path"] = path
        self.lineage["download_timestamp"] = datetime.now().isoformat()
        self.logger.info("Dataset downloaded (or confirmed present) at %s", path)
        return path

    def get_data(self, force_download: bool = False) -> pd.DataFrame:
        """
        Ensure data is available, then load into a pandas DataFrame.

        If the main CSV file is not present (or `force_download=True`),
        the downloader will be invoked.

        Returns
        -------
        pd.DataFrame
            Loaded dataset with standardized feature names and target column.
        """
        data_dir = getattr(self.config, "data_dir", "data")
        data_file = os.path.join(data_dir, "MiniBooNE_PID.csv")

        if force_download or not os.path.exists(data_file):
            self.logger.info("Data file missing or forced re-download requested.")
            data_file = self.download()

        self.logger.info("Loading MiniBooNE Dataset")
        self.df = self.loader.load(data_file)

        self.lineage["data_path"] = data_file
        self.lineage["data_hash"] = self._compute_dataset_hash(self.df)
        self.lineage["load_timestamp"] = datetime.now().isoformat()

        return self.df

    def load(self) -> pd.DataFrame:
        """
        Load an existing dataset from disk without triggering download.

        Expects the file `MiniBooNE_PID.csv` in `config.data_dir`.

        Returns
        -------
        pd.DataFrame
            Loaded dataset.
        """
        data_dir = getattr(self.config, "data_dir", "data")
        data_file = os.path.join(data_dir, "MiniBooNE_PID.csv")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found at {data_file}")

        self.logger.info("Loading dataset directly from %s", data_file)
        self.df = self.loader.load(data_file)

        self.lineage["data_path"] = data_file
        self.lineage["data_hash"] = self._compute_dataset_hash(self.df)
        self.lineage["load_timestamp"] = datetime.now().isoformat()

        return self.df

    # -------------------------------------------------------------------------
    # CLEANING
    # -------------------------------------------------------------------------
    def clean_data(self) -> pd.DataFrame:
        """
        Apply data cleaning (duplicates, NaNs, -999 placeholders).

        Returns
        -------
        pd.DataFrame
            Cleaned dataset.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call get_data() or load() first.")

        self.logger.info("Starting data cleaning...")
        self.df = self.cleaner.clean(self.df)

        self.lineage["clean_timestamp"] = datetime.now().isoformat()
        self.lineage["clean_summary"] = (
            "Duplicates removed, NaNs handled, -999 replaced (see DataCleaner)"
        )

        self.logger.info("Data cleaning complete: shape=%s", self.df.shape)
        return self.df

    # -------------------------------------------------------------------------
    # PREPROCESSING / PROCESSING
    # -------------------------------------------------------------------------
    def preprocess(
        self,
        df: Optional[pd.DataFrame] = None,
        to_numpy: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Run preprocessing on an already-cleaned DataFrame (no splitting).

        This is mainly useful for:
        - Applying a fitted pipeline to new data (inference)
        - Transforming a subset for analysis

        Parameters
        ----------
        df : Optional[pd.DataFrame]
            Cleaned input data. If None, uses `self.df`.
        to_numpy : bool, default=True
            If True, returns a NumPy array; otherwise, a DataFrame.

        Returns
        -------
        Union[pd.DataFrame, np.ndarray]
            Preprocessed features.
        """
        if df is None:
            if self.df is None:
                raise ValueError("No dataset available. Load and clean data first.")
            df = self.df

        X, _ = self.processor.prepare_features(df)
        X_proc = self.processor.run_pipeline(X)

        self.logger.info("Preprocessing applied: shape=%s", X_proc.shape)
        return X_proc.to_numpy() if to_numpy else X_proc

    def process(
        self,
        clean: bool = True,
        export_pipeline: bool = False,
    ) -> Dict[str, Tuple[np.ndarray, pd.Series]]:
        """
        Execute the full pipeline:
        - Optional cleaning
        - Preprocessing (scaling, filtering)
        - Train/test splitting
        - Optional pipeline export

        Parameters
        ----------
        clean : bool, default=True
            Whether to run cleaning before preprocessing.
        export_pipeline : bool, default=False
            Whether to export the fitted preprocessing pipeline.

        Returns
        -------
        Dict[str, Tuple[np.ndarray, pd.Series]]
            Dictionary of train/test splits (NumPy-based).
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call get_data() or load() first.")

        if clean:
            self.logger.info("Cleaning data before processing...")
            self.clean_data()

        self.logger.info("Running DataProcessor pipeline...")
        X_train_df, X_test_df, y_train, y_test = self.processor.process(self.df, to_numpy=False)
        X_train_np, X_test_np = X_train_df.to_numpy(), X_test_df.to_numpy()

        # Store both formats
        self.splits_df = {
            "train": (X_train_df, y_train),
            "test": (X_test_df, y_test),
        }
        self.splits = {
            "train": (X_train_np, y_train),
            "test": (X_test_np, y_test),
        }

        if export_pipeline:
            self.processor.export_pipeline()
            self.lineage["pipeline_exported"] = True
        else:
            self.lineage["pipeline_exported"] = False

        self.lineage["process_timestamp"] = datetime.now().isoformat()

        self.logger.info(
            "Processing complete. Train: X=%s, Test: X=%s",
            X_train_df.shape,
            X_test_df.shape,
        )
        return self.splits

    # -------------------------------------------------------------------------
    # SPLIT & DATA UTILITIES
    # -------------------------------------------------------------------------
    def get_splits(
        self, as_dataframe: bool = False
    ) -> Mapping[str, Tuple[Union[np.ndarray, pd.DataFrame], pd.Series]]:
        """
        Retrieve train/test splits in the desired format.

        Parameters
        ----------
        as_dataframe : bool, default=False
            If True, return DataFrame-based splits; otherwise NumPy-based.

        Returns
        -------
        Dict[str, Tuple[Union[np.ndarray, pd.DataFrame], pd.Series]]
            A mapping from split name to (X, y) pair.

        Raises
        ------
        ValueError
            If splits have not been computed yet.
        """
        if as_dataframe:
            if not self.splits_df:
                raise ValueError("No DataFrame splits available. Run process() first.")
            return self.splits_df  # type: ignore

        if not self.splits:
            raise ValueError("No NumPy splits available. Run process() first.")
        return self.splits

    def get_feature_names(self) -> Tuple[str, ...]:
        """
        Return the feature column names (excluding the target).

        Returns
        -------
        Tuple[str, ...]
            Names of feature columns.

        Raises
        ------
        ValueError
            If data has not been loaded.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call get_data() or load() first.")

        target_col = getattr(self.config, "target_col", "signal")
        return tuple(col for col in self.df.columns if col != target_col)

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get a compact summary of the loaded dataset.

        Returns
        -------
        Dict[str, Any]
            Various statistics such as sample counts, class balance,
            feature count, and missing value count.

        Raises
        ------
        ValueError
            If data has not been loaded.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call get_data() or load() first.")

        target_col = getattr(self.config, "target_col", "signal")

        return {
            "total_samples": len(self.df),
            "signal_count": int((self.df[target_col] == 1).sum()),
            "background_count": int((self.df[target_col] == 0).sum()),
            "signal_ratio": float((self.df[target_col] == 1).mean()),
            "feature_count": len(self.df.columns) - 1,  # exclude target
            "missing_values": int(self.df.isnull().sum().sum()),
        }

    def get_split_shapes(self) -> Dict[str, Tuple[int, int]]:
        """
        Return the (n_samples, n_features) shapes for processed splits.

        Returns
        -------
        Dict[str, Tuple[int, int]]
            Mapping from split name to (rows, cols) of X.

        Raises
        ------
        ValueError
            If splits are not available.
        """
        if not self.splits:
            raise ValueError("No splits available. Run process() first.")

        shapes: Dict[str, Tuple[int, int]] = {}
        for name, (X, _) in self.splits.items():
            shapes[name] = (int(X.shape[0]), int(X.shape[1]))
        return shapes

    def save_processed_data(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Save processed train/test splits to disk as CSV files.

        Useful for debugging, sharing preprocessed datasets, or inspecting
        intermediate results outside the Python environment.

        Parameters
        ----------
        output_dir : Optional[Union[str, Path]], default=None
            Directory to save processed splits. Defaults to `config.data_dir/processed`.

        Raises
        ------
        ValueError
            If splits_df are not available (process() not run).
        """
        if not self.splits_df:
            raise ValueError("No processed splits available. Run process() first.")

        base_dir = Path(
            output_dir or (Path(getattr(self.config, "data_dir", "data")) / "processed")
        )
        base_dir.mkdir(parents=True, exist_ok=True)

        for split_name, (X_df, y) in self.splits_df.items():
            X_path = base_dir / f"X_{split_name}.csv"
            y_path = base_dir / f"y_{split_name}.csv"
            X_df.to_csv(X_path, index=False)
            y.to_csv(y_path, index=False)
            self.logger.info("Saved processed split '%s' to %s and %s", split_name, X_path, y_path)

    # -------------------------------------------------------------------------
    # LINEAGE / SUMMARY
    # -------------------------------------------------------------------------
    def export_lineage(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Export lineage metadata as JSON for scientific reproducibility.

        Parameters
        ----------
        path : Optional[Union[str, Path]], default=None
            Destination path for lineage JSON. Defaults to
            `config.cache_dir / 'lineage.json'`.

        Returns
        -------
        Path
            Path where lineage was written.
        """
        lineage_path = Path(
            path or (Path(getattr(self.config, "cache_dir", "data/processed")) / "lineage.json")
        )
        lineage_path.parent.mkdir(parents=True, exist_ok=True)

        with open(lineage_path, "w") as f:
            json.dump(self.lineage, f, indent=2, default=str)

        self.logger.info("Lineage exported to %s", lineage_path)
        return lineage_path

    @staticmethod
    def _compute_dataset_hash(df: pd.DataFrame) -> str:
        """
        Compute a SHA-1 hash for the DataFrame contents.

        Used to track data provenance and detect unintended changes.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataset.

        Returns
        -------
        str
            Hex-encoded SHA-1 hash.
        """
        data_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
        return hashlib.sha1(data_bytes).hexdigest()

    def summary(self) -> None:
        """
        Log a human-readable summary of the current dataset, splits, and lineage.

        This is intended for quick inspection in notebooks and logs, not as
        a machine-readable API (use get_data_summary/get_split_shapes instead).
        """
        if self.df is None:
            self.logger.warning("No dataset loaded.")
            return

        self.logger.info("Dataset shape: %s", self.df.shape)
        target_col = getattr(self.config, "target_col", "signal")
        self.logger.info("Class distribution:\n%s", self.df[target_col].value_counts())

        if self.splits_df:
            for name, (X, y) in self.splits_df.items():
                self.logger.info("Split '%s': X=%s, y=%s", name, X.shape, y.shape)

        if self.lineage:
            self.logger.info("Lineage metadata:")
            for key, value in self.lineage.items():
                self.logger.info("  %s: %s", key, value)

    def __repr__(self) -> str:
        n_rows = len(self.df) if self.df is not None else 0
        n_cols = len(self.df.columns) if self.df is not None else 0
        return f"<MiniBooNEDataHandler rows={n_rows} cols={n_cols}>"
