# src/data/data_handler.py
import os
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kaggle.api.kaggle_api_extended import KaggleApi
from typing import Dict, Tuple, Optional
import numpy as np

from ..config import DataConfig


class MiniBooNEDataHandler:
    """Handles downloading, loading, and preprocessing of the MiniBooNE dataset."""

    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.data_dir = self.config.data_dir
        self.data_file = os.path.join(self.data_dir, "MiniBooNE_PID.csv")
        self.scaler = StandardScaler()
        self.df: Optional[pd.DataFrame] = None
        self.splits: Dict[str, Tuple[np.ndarray, pd.Series]] = {}

        os.makedirs(self.data_dir, exist_ok=True)

    def download(self) -> None:
        """Download dataset from Kaggle if not found locally."""
        if os.path.exists(self.data_file):
            print("✅ Dataset already exists locally.")
            return

        print("⬇️  Downloading dataset from Kaggle...")
        try:
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(self.config.dataset, path=self.data_dir, unzip=True)

            # Unzip and cleanup
            for f in os.listdir(self.data_dir):
                if f.endswith(".zip"):
                    with zipfile.ZipFile(os.path.join(self.data_dir, f), "r") as zf:
                        zf.extractall(self.data_dir)
                    os.remove(os.path.join(self.data_dir, f))
            print("✅ Download complete.")

        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")

    def load(self) -> pd.DataFrame:
        """Load the dataset into a pandas DataFrame."""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"{self.data_file} not found. Run download() first.")

        print("📂 Loading data...")
        self.df = pd.read_csv(self.data_file)

        self.df["signal"] = (self.df.index < (self.config.number_of_signals)).astype(int)

        # Validate dataset structure
        self._validate_dataset()

        print(f"✅ Loaded {len(self.df)} rows, {len(self.df.columns)} columns.")
        print(f"📊 Signal events: {(self.df['signal'] == 1).sum():,}")
        print(f"📊 Background events: {(self.df['signal'] == 0).sum():,}")

        return self.df

    def _validate_dataset(self) -> None:
        """Validate the dataset structure and contents."""
        if self.df is None:
            raise ValueError("No data loaded")

        if "signal" not in self.df.columns:
            raise ValueError("Dataset must contain 'signal' column")

        # Check if signal counts match expected values
        signal_count = (self.df["signal"] == 1).sum()
        background_count = (self.df["signal"] == 0).sum()

        print(
            f"🔍 Validation: {signal_count:,} signal vs expected {self.config.number_of_signals:,}"
        )
        print(
            f"🔍 Validation: {background_count:,} background vs expected {self.config.number_of_background:,}"
        )

        if signal_count != (self.config.number_of_signals):
            print(
                f"⚠️  Warning: Signal count mismatch. Expected {self.config.number_of_signals:,}, got {signal_count:,}"
            )

        if background_count != self.config.number_of_background:
            print(
                f"⚠️  Warning: Background count mismatch. Expected {self.config.number_of_background:,}, got {background_count:,}"
            )

        if (signal_count + background_count) != len(self.df):
            print(
                f"⚠️  Warning: Total count mismatch. Expected {self.config.number_of_background+self.config.number_of_signals}, got {len(self.df)}"
            )
        else:
            print("✅ Total number of events corretly imported")

    def clean_data(self) -> pd.DataFrame:
        """Clean the dataset for missing values, duplicates and general stats per column"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        if self.df.isnull().sum().sum() != 0:
            print(f"There are in total: {self.df.isnull().sum().sum()} missing values")
        else:
            print("✅ There are no missing values")

        if self.df.duplicated().sum() != 0:
            print(f"There are in total {self.df.duplicated().sum()} duplicated values")
            self.df = self.df.drop_duplicates()
            print("✅ Dropped the duplicated values")
        else:
            print("✅ There are no duplicated values")

        return self.df

    def preprocess(self) -> Dict[str, Tuple[np.ndarray, pd.Series]]:
        """Scale and split data into train/val/test sets."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        print("🔄 Preprocessing data...")

        X = self.df.drop("signal", axis=1)
        y = self.df["signal"]

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

        self.splits = {
            "train": (X_train_scaled, y_train.reset_index(drop=True)),
            "val": (X_val_scaled, y_val.reset_index(drop=True)),
            "test": (X_test_scaled, y_test.reset_index(drop=True)),
            "raw": {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)},
        }

        # Print split statistics
        self._print_split_stats()
        print("✅ Preprocessing complete.")

        return self.splits

    def _print_split_stats(self) -> None:
        """Print statistics about the data splits."""
        if not self.splits:
            return

        print("📊 Data Split Statistics:")
        for split_name, (X, y) in self.splits.items():
            if split_name != "raw":  # Skip raw data in stats
                signal_pct = (y == 1).mean() * 100
                print(
                    f"   {split_name.capitalize()}: {len(X):,} samples ({signal_pct:.1f}% signal)"
                )

    def get_feature_names(self) -> list:
        """Get the names of the features (excluding target)."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")
        return [col for col in self.df.columns if col != "signal"]

    def get_data_summary(self) -> Dict[str, any]:
        """Get a summary of the dataset."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load() first.")

        return {
            "total_samples": len(self.df),
            "signal_count": (self.df["signal"] == 1).sum(),
            "background_count": (self.df["signal"] == 0).sum(),
            "signal_ratio": (self.df["signal"] == 1).mean(),
            "feature_count": len(self.df.columns) - 1,  # exclude target
            "feature_names": self.get_feature_names(),
            "missing_values": self.df.isnull().sum().sum(),
        }

    def save_processed_data(self, output_dir: str = "processed_data") -> None:
        """Save processed splits to disk."""
        if not self.splits:
            raise ValueError("No processed data available. Call preprocess() first.")

        os.makedirs(output_dir, exist_ok=True)

        for split_name, (X, y) in self.splits.items():
            if split_name != "raw":  # Don't save raw data
                # Save as numpy arrays
                np.save(os.path.join(output_dir, f"X_{split_name}.npy"), X)
                np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y.values)

        # Save feature names and scaler
        import joblib

        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        joblib.dump(self.get_feature_names(), os.path.join(output_dir, "feature_names.pkl"))

        print(f"💾 Processed data saved to {output_dir}/")
