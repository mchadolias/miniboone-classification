"""
Physics-aware, config-driven preprocessing for MiniBooNE neutrino data.
Handles scaling, filtering, optional outlier detection, and caching.

Supports:
- Optional NumPy conversion
- sklearn Pipeline export / loading
- Reusable transform-only path for inference
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

from src.config.config import DataConfig


class DataProcessor:
    """
    Configurable preprocessing pipeline for the MiniBooNE dataset.

    Responsibilities
    ----------------
    - Extracts features and target from the cleaned DataFrame
    - Applies variance filtering and scaling via sklearn Pipeline
    - Optionally adds an outlier flag based on z-score
    - Splits into stratified train/test sets
    - Supports caching of processed splits
    - Exports / loads the fitted preprocessing pipeline for reuse

    Parameters
    ----------
    config : DataConfig
        Configuration controlling target column name, scaling method,
        variance threshold, test size, random state, caching behavior, etc.

    Attributes
    ----------
    config : DataConfig
        Active configuration.
    pipeline : Optional[Pipeline]
        Fitted sklearn pipeline (variance filter + scaler).
    feature_names_ : Optional[List[str]]
        Names of features retained after variance filtering.
    cache_dir : Path
        Directory used to store cached processed data and metadata.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.pipeline: Optional[Pipeline] = None
        self.feature_names_: Optional[List[str]] = None

        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    # ======================================================================
    # MAIN ENTRY POINT
    # ======================================================================
    def process(
        self,
        df: pd.DataFrame,
        to_numpy: bool = False,
    ) -> Tuple[
        Union[pd.DataFrame, np.ndarray],
        Union[pd.DataFrame, np.ndarray],
        pd.Series,
        pd.Series,
    ]:
        """
        Run the full preprocessing pipeline on the given DataFrame.

        Steps
        -----
        1. Compute a cache key based on config + data preview.
        2. Load from cache if valid and `config.use_cache` is True.
        3. Otherwise:
           - Extract features/target
           - Fit & apply sklearn pipeline (variance filter + scaler)
           - Optionally add an outlier flag
           - Stratified train/test split
           - Save to cache
        4. Optionally convert X splits to NumPy arrays.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataset including the target column.
        to_numpy : bool, default=False
            If True, X_train and X_test are returned as np.ndarray.
            If False, they are returned as pd.DataFrame.

        Returns
        -------
        Tuple[Union[pd.DataFrame, np.ndarray],
              Union[pd.DataFrame, np.ndarray],
              pd.Series,
              pd.Series]
            X_train, X_test, y_train, y_test
        """
        self.logger.info("Starting data processing pipeline...")
        t0 = time.time()

        cache_key = self._get_cache_key(df)
        cache_paths = self._get_cache_paths(cache_key)

        if self.config.use_cache and self._is_cache_valid(cache_paths):
            self.logger.info("Loading processed data from cache...")
            X_train, X_test, y_train, y_test = self._load_cache(cache_paths)
        else:
            # Fresh processing
            X, y = self.prepare_features(df)
            X_processed = self.run_pipeline(X, fit=True)

            if self.config.add_outlier_flag:
                X_processed = self.add_outlier_flag(X_processed)

            X_train, X_test, y_train, y_test = self.split_data(X_processed, y)
            self.save_to_cache(cache_paths, X_train, X_test, y_train, y_test)
            self.logger.info(f"Processed data cached under key: {cache_key[:8]}")

        if to_numpy:
            X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
            X_test = X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test

        self.logger.info("Processing complete in %.2fs", time.time() - t0)
        return X_train, X_test, y_train, y_test

    # ======================================================================
    # PIPELINE STAGES
    # ======================================================================
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract feature matrix X and target vector y from a cleaned DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned dataset.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            (X, y) where X contains only feature columns and y is the target.

        Raises
        ------
        ValueError
            If the target column is not present.
        """
        target = self.config.target_col
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame.")

        X = df.drop(columns=[target])
        y = df[target].astype(int)
        self.logger.debug("Prepared features X%s and target y%s", X.shape, y.shape)
        return X, y

    def run_pipeline(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Build (if needed) and apply the sklearn preprocessing pipeline.

        By default, this method FITS and TRANSFORMS the data (fit=True).
        When called with fit=False, it only applies the existing fitted
        pipeline (transform-only) and does not refit.

        Parameters
        ----------
        X : pd.DataFrame
            Input feature matrix.
        fit : bool, default=True
            If True, creates/fits a new pipeline on X.
            If False, expects an existing fitted pipeline and only transforms X.

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix as a DataFrame.

        Raises
        ------
        ValueError
            If fit=False but no pipeline has been fitted yet.
        """
        if fit:
            scaler = (
                RobustScaler()
                if getattr(self.config, "scale_method", "standard") == "robust"
                else StandardScaler()
            )
            pipeline_steps = [
                ("variance_filter", VarianceThreshold(threshold=self.config.variance_threshold)),
                ("scaler", scaler),
            ]
            self.pipeline = Pipeline(pipeline_steps)

            self.logger.debug("Fitting pipeline with steps: %s", [n for n, _ in pipeline_steps])
            X_transformed = self.pipeline.fit_transform(X)

            # Store feature names after variance filter
            support = self.pipeline.named_steps["variance_filter"].get_support()
            self.feature_names_ = X.columns[support].tolist()
        else:
            if self.pipeline is None:
                raise ValueError(
                    "Pipeline has not been fitted. Call run_pipeline(..., fit=True) or process() first."
                )
            self.logger.debug("Applying transform-only pipeline to X%s", X.shape)
            X_transformed = self.pipeline.transform(X)

            # If feature_names_ is missing for some reason, fall back gracefully
            if self.feature_names_ is None:
                # Best-effort: assume all columns were retained
                self.feature_names_ = list(X.columns)

        X_processed = pd.DataFrame(X_transformed, columns=self.feature_names_, index=X.index)
        self.logger.info("Preprocessing produced matrix of shape %s", X_processed.shape)
        return X_processed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the existing fitted pipeline to new data without refitting.

        This method is intended for inference-time transformations, where the
        preprocessing pipeline has already been fitted on training data.

        Parameters
        ----------
        X : pd.DataFrame
            New feature matrix to transform.

        Returns
        -------
        pd.DataFrame
            Transformed features as a DataFrame.

        Raises
        ------
        ValueError
            If the pipeline has not been fitted.
        """
        return self.run_pipeline(X, fit=False)

    def add_outlier_flag(self, X: pd.DataFrame, z_thresh: float = 4.0) -> pd.DataFrame:
        """
        Add an 'is_outlier' flag column based on a z-score threshold.

        Any row with at least one feature whose absolute z-score exceeds
        `z_thresh` is flagged as an outlier.

        Parameters
        ----------
        X : pd.DataFrame
            Preprocessed feature matrix.
        z_thresh : float, default=4.0
            Z-score threshold for flagging outliers.

        Returns
        -------
        pd.DataFrame
            Feature matrix with an extra 'is_outlier' column.
        """
        z_scores = np.abs((X - X.mean()) / X.std(ddof=0))
        X = X.copy()
        X["is_outlier"] = (z_scores > z_thresh).any(axis=1).astype(int)
        self.logger.info("Flagged %d outliers (z > %.1f).", int(X["is_outlier"].sum()), z_thresh)
        return X

    def split_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Stratified train/test split of features and target.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        self.logger.info(
            "Data split into train=%s and test=%s",
            X_train.shape,
            X_test.shape,
        )
        return X_train, X_test, y_train, y_test

    # ======================================================================
    # PIPELINE EXPORT / IMPORT
    # ======================================================================
    def export_pipeline(self, filename: str = "preprocessing_pipeline.joblib") -> Path:
        """
        Export the fitted sklearn preprocessing pipeline to disk.

        Parameters
        ----------
        filename : str, default="preprocessing_pipeline.joblib"
            Name of the output file (stored in config.cache_dir).

        Returns
        -------
        Path
            Path to the exported pipeline file.

        Raises
        ------
        ValueError
            If the pipeline has not been fitted yet.
        """
        if self.pipeline is None:
            raise ValueError(
                "Pipeline not initialized. Run process() or run_pipeline(..., fit=True) first."
            )

        path = Path(self.config.cache_dir) / filename
        joblib.dump(self.pipeline, path)
        self.logger.info("Preprocessing pipeline exported to %s", path)
        return path

    def load_pipeline(self, filename: str = "preprocessing_pipeline.joblib") -> Optional[Pipeline]:
        """
        Load a previously exported preprocessing pipeline.

        Parameters
        ----------
        filename : str, default="preprocessing_pipeline.joblib"
            Name of the pipeline file inside config.cache_dir.

        Returns
        -------
        Optional[Pipeline]
            The loaded sklearn Pipeline instance.

        Raises
        ------
        FileNotFoundError
            If the pipeline file is missing.
        """
        path = Path(self.config.cache_dir) / filename
        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found at {path}")

        self.pipeline = joblib.load(path)
        self.logger.info("Preprocessing pipeline loaded from %s", path)
        return self.pipeline

    # ======================================================================
    # CACHING UTILITIES
    # ======================================================================
    def _get_cache_key(self, df: pd.DataFrame) -> str:
        """
        Create a cache key based on a preview of the data and the config.

        Uses:
        - JSON dump of the first 100 rows (values only)
        - JSON dump of the config.__dict__ (sorted keys)
        """
        preview = df.head(100).to_json(orient="split")
        params = json.dumps(self.config.__dict__, sort_keys=True, default=str)
        return hashlib.md5((params + preview).encode()).hexdigest()

    def _get_cache_paths(self, key: str) -> Dict[str, Path]:
        """
        Return a dictionary of paths for cached train/test splits and metadata.
        """
        base = self.cache_dir / f"processed_{key}"
        return {
            "X_train": base.with_suffix(".X_train.parquet"),
            "X_test": base.with_suffix(".X_test.parquet"),
            "y_train": base.with_suffix(".y_train.parquet"),
            "y_test": base.with_suffix(".y_test.parquet"),
            "meta": base.with_suffix(".json"),
        }

    def _is_cache_valid(self, paths: Dict[str, Path]) -> bool:
        """
        Check whether all required cache files exist.
        """
        valid = all(Path(p).exists() for p in paths.values())
        if not valid:
            self.logger.debug("Cache invalid or incomplete — will recompute.")
        return valid

    def save_to_cache(
        self,
        paths: Dict[str, Path],
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> None:
        """
        Save processed splits and metadata to cache.
        """
        X_train.to_parquet(paths["X_train"])
        X_test.to_parquet(paths["X_test"])
        y_train.to_frame("target").to_parquet(paths["y_train"])
        y_test.to_frame("target").to_parquet(paths["y_test"])

        metadata = {
            "created_at": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "features": list(X_train.columns),
        }
        with open(paths["meta"], "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        self.logger.debug("Cache metadata saved to %s", paths["meta"])

    def _load_cache(self, paths: Dict[str, Path]):
        """
        Load cached train/test splits.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            X_train, X_test, y_train, y_test
        """
        X_train = pd.read_parquet(paths["X_train"])
        X_test = pd.read_parquet(paths["X_test"])
        y_train = pd.read_parquet(paths["y_train"])["target"]
        y_test = pd.read_parquet(paths["y_test"])["target"]
        self.logger.info("Loaded cached splits: train=%s, test=%s", X_train.shape, X_test.shape)
        return X_train, X_test, y_train, y_test
