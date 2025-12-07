"""
Enhanced data cleaning module for MiniBooNE particle classification dataset.

Handles the specific challenges of the MiniBooNE dataset:
- Extremely skewed distributions
- Heavy tails and outliers
- Preserving the 70/30 background-signal ratio
- Unknown, but physical meaning of features (particle detection measurements)
"""

from typing import Dict

import numpy as np
import pandas as pd
import logging


class DataCleaner:
    """
    Enhanced data cleaner for MiniBooNE dataset with specialized handling
    for skewed distributions and outliers while preserving signal structure.

    Attributes
    ----------
    target_col : str
        Name of the target column in the dataset.
    cleaning_stats : Dict
        Statistics collected during cleaning for reporting.
    outlier_thresholds : Dict
        Thresholds used for outlier detection per feature.
    transformation_params : Dict
        Parameters for any transformations applied to features.
    """

    def __init__(self, config=None, logger=None):
        self.config = config or {}
        self.target_col: str = "signal"
        self.cleaning_stats = {}
        self.outlier_thresholds = {}
        self.transformation_params = {}
        self.logger = logger or logging.getLogger(__name__)

    def _handle_duplicated_events(self, df: pd.DataFrame) -> pd.DataFrame:
        original_size = len(df)
        df_no_dupes = df.drop_duplicates()

        if len(df_no_dupes) < original_size:
            n_removed = original_size - len(df_no_dupes)
            self.logger.info(f"Removed {n_removed} duplicated events from dataset.")

        return df_no_dupes  # Returns cleaned DataFrame

    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing or invalid data while preserving physical events.

        This method:
        1. Replaces MiniBooNE sentinel values (-999) with NaN.
        2. Performs median imputation for numeric columns.
        3. Handles fully missing columns using a global fallback median.
        4. Logs detailed information about missing data patterns.

        Args:
            df (pd.DataFrame): Input DataFrame possibly containing NaNs or -999 placeholders.

        Returns:
            pd.DataFrame: Cleaned DataFrame with missing values imputed.
        """
        df_clean = df.copy()

        sentinel_mask = df_clean == -999
        sentinel_count = sentinel_mask.sum().sum()
        if sentinel_count > 0:
            self.logger.info(
                f"Detected {int(sentinel_count)} sentinel (-999) entries; replacing with NaN."
            )
            df_clean.replace(-999, np.nan, inplace=True)
        else:
            self.logger.debug("No sentinel (-999) placeholders detected.")

        missing_summary = df_clean.isnull().sum()
        total_missing = missing_summary.sum()

        if total_missing == 0:
            self.logger.info("No missing values detected — returning original DataFrame.")
            return df_clean

        self.logger.info(
            f"Detected {int(total_missing)} missing entries across "
            f"{(missing_summary > 0).sum()} columns."
        )

        global_median = df_clean.median(numeric_only=True).median()

        for col, n_missing in missing_summary.items():
            if n_missing == 0:
                continue

            col_dtype = df_clean[col].dtype
            if pd.api.types.is_numeric_dtype(col_dtype):
                median_val = df_clean[col].median()
                if np.isnan(median_val):
                    # Entire column missing — use fallback global median
                    fallback_val = global_median if not np.isnan(global_median) else 0.0
                    self.logger.warning(
                        f"Column '{col}' entirely missing; filling with fallback value {fallback_val:.4f}."
                    )
                    median_val = fallback_val
                df_clean[col] = df_clean[col].fillna(median_val)
                self.logger.debug(
                    f"Filled {n_missing} NaNs in '{col}' with median ({median_val:.4f})."
                )
            else:
                self.logger.warning(
                    f"Column '{col}' is non-numeric and contains {n_missing} missing values. "
                    "Leaving as-is. Consider manual review."
                )

        remaining = df_clean.isnull().sum().sum()
        if remaining > 0:
            self.logger.warning(f"{int(remaining)} missing values remain after imputation.")
        else:
            self.logger.info("All missing values successfully imputed with column medians.")

        return df_clean

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Minimal cleaning - preserve all physical events, handle actual data issues
        """
        df_clean = df.copy()

        # 1. Remove actual duplicates (measurement errors)
        df_clean = self._handle_duplicated_events(df_clean)

        # 2. Handle missing values (instrumentation issues)
        df_clean = self._handle_missing_data(df_clean)

        # 3. Analyze distributions for scaling recommendations
        self._analyze_distributions(df_clean, self.target_col)

        return df_clean

    def _analyze_distributions(self, df: pd.DataFrame, target_col: str) -> None:
        """Analyze feature distributions and recommend appropriate scaling"""
        feature_cols = [col for col in df.columns if col != target_col]

        scaling_recommendations = {}

        for feature in feature_cols:
            data = df[feature].dropna()
            if len(data) == 0:
                continue

            # Get scaling recommendation based on data characteristics
            scale_type = self._recommend_scaling_type(data)
            scaling_recommendations[feature] = {
                "recommended_scale": scale_type,
                "min": data.min(),
                "max": data.max(),
                "dynamic_range": self._calculate_dynamic_range(data),
                "skewness": data.skew(),
                "has_negatives": (data < 0).any(),
                "has_positives": (data > 0).any(),
                "span_zero": (data.min() <= 0 <= data.max()),
            }

        self.cleaning_stats["scaling_recommendations"] = scaling_recommendations

        # Print scaling recommendations
        self._print_scaling_summary(scaling_recommendations)

    def _recommend_scaling_type(self, data: pd.Series) -> str:
        """
        Recommend appropriate scaling type based on data distribution.

        Returns:
            str: 'linear', 'symlog', or 'asinh'
        """
        if len(data) < 2:
            return "linear"

        has_negatives = (data < 0).any()
        has_positives = (data > 0).any()
        dynamic_range = self._calculate_dynamic_range(data)

        # Case 1: Only positive values - can use log scale
        if has_positives and not has_negatives:
            positive_data = data[data > 0]
            if len(positive_data) > 1:
                pos_range = positive_data.max() / positive_data.min()
                skewness = float(data.skew())
                if pos_range > 1000 or abs(skewness) > 3:
                    return "log"

        # Case 2: Mixed negative and positive values
        elif has_negatives and has_positives:
            # Use symlog for large dynamic ranges spanning zero
            if dynamic_range > 1e6:  # Very large dynamic range
                return "symlog"
            elif dynamic_range > 1e4:  # Large dynamic range
                return "asinh"  # Alternative: inverse hyperbolic sine
        # Case 3: Only negative values (less common)
        elif has_negatives and not has_positives:
            negative_data = data[data < 0]
            if len(negative_data) > 1:
                neg_range = abs(negative_data.min() / negative_data.max())
                skewness = float(data.skew())
                if neg_range > 1000 or abs(skewness) > 3:
                    return "symlog"

        return "linear"

    def _calculate_dynamic_range(self, data: pd.Series) -> float:
        """
        Calculate dynamic range appropriately for data spanning negative/positive.
        """
        abs_max = max(abs(data.min()), abs(data.max()))
        # For data spanning zero, use smallest non-zero absolute value
        nonzero_abs = data[data != 0].abs()
        if len(nonzero_abs) > 0:
            abs_min = nonzero_abs.min()
            return abs_max / abs_min if abs_min > 0 else float("inf")
        return 1.0

    def _print_scaling_summary(self, recommendations: Dict) -> None:
        """Print summary of scaling recommendations"""
        scale_counts = {}
        for feat, rec in recommendations.items():
            scale_type = rec["recommended_scale"]
            scale_counts[scale_type] = scale_counts.get(scale_type, 0) + 1

        self.logger.info("Feature Scaling Recommendations Summary:")
        for scale_type, count in scale_counts.items():
            features = [
                f for f, rec in recommendations.items() if rec["recommended_scale"] == scale_type
            ]
            self.logger.info(f" - {scale_type}: {count} features")

            # Show examples of each type
            if features:
                examples = features[:3]  # Show first 3 examples
                ranges = [
                    f"{recommendations[f]['min']:.1e} to {recommendations[f]['max']:.1e}"
                    for f in examples
                ]
                self.logger.info(
                    f"   Examples: {', '.join(examples)} with ranges {', '.join(ranges)}"
                )
