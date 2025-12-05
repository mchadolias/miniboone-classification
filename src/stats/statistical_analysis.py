# src/stats/statistical_analysis.py

import numpy as np
import pandas as pd
from typing import Literal, Optional, List
from scipy.stats import mannwhitneyu, ks_2samp, ttest_ind, entropy
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests
from src.utils.logger import get_module_logger

logger = get_module_logger(__name__)


# -------------------------------------------------------------------------
# Bootstrap Error Estimation
# -------------------------------------------------------------------------
def compute_bootstrap_error(
    df_column: pd.Series,
    label_value: str,
    n_boot: int = 1000,
    stat: Literal["count", "percent"] = "count",
) -> float:
    """Estimate bootstrap standard error for a categorical label.

    Args:
        df_column (pd.Series): Series of categorical or binary values.
        label_value (str): Category to estimate the error for.
        n_boot (int, optional): Number of bootstrap resamples. Defaults to 1000.
        stat (Literal["count", "percent"], optional): Whether to compute
            standard deviation of raw counts or normalized percentages.

    Returns:
        float: Bootstrapped standard deviation for the label estimate.
    """
    if df_column.empty:
        logger.warning("Empty column passed to compute_bootstrap_error(). Returning NaN.")
        return np.nan

    samples = []
    for _ in range(n_boot):
        resample = df_column.sample(frac=1, replace=True)
        val = resample.value_counts(normalize=(stat == "percent")).get(label_value, 0)
        samples.append(val)
    return float(np.std(samples))


# -------------------------------------------------------------------------
# Bootstrap Confidence Interval
# -------------------------------------------------------------------------
def bootstrap_ci(
    data: pd.Series,
    stat: Literal["mean", "median"] = "mean",
    n_boot: int = 1000,
    ci: float = 95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data (pd.Series): Numeric data.
        stat (Literal["mean", "median"], optional): Statistic to estimate.
        n_boot (int, optional): Number of bootstrap resamples. Defaults to 1000.
        ci (float, optional): Confidence interval percentage. Defaults to 95.

    Returns:
        tuple[float, float]: Lower and upper confidence interval bounds.
    """
    data = data.dropna()
    if data.empty:
        return np.nan, np.nan

    values = []
    for _ in range(n_boot):
        sample = data.sample(frac=1, replace=True)
        values.append(getattr(sample, stat)())
    lower = np.percentile(values, (100 - ci) / 2)
    upper = np.percentile(values, 100 - (100 - ci) / 2)
    return lower, upper


# -------------------------------------------------------------------------
# Effect Size Metrics
# -------------------------------------------------------------------------
def compute_effect_size(
    group1: pd.Series,
    group2: pd.Series,
    method: Literal["cohen_d", "cliffs_delta"] = "cohen_d",
) -> Optional[float]:
    """Compute standardized effect size between two distributions.

    Args:
        group1 (pd.Series): First sample group (e.g., background).
        group2 (pd.Series): Second sample group (e.g., signal).
        method (Literal["cohen_d", "cliffs_delta"], optional): Effect size type.

    Returns:
        Optional[float]: Effect size value, or NaN if invalid input.
    """
    g1, g2 = group1.dropna(), group2.dropna()
    if g1.empty or g2.empty:
        return np.nan

    if method == "cohen_d":
        pooled_std = np.sqrt(((g1.var() + g2.var()) / 2))
        return (g1.mean() - g2.mean()) / pooled_std if pooled_std > 0 else np.nan
    elif method == "cliffs_delta":
        n1, n2 = len(g1), len(g2)
        delta = sum((x > y) - (x < y) for x in g1 for y in g2) / (n1 * n2)
        return delta
    else:
        raise ValueError(f"Unknown method: {method}")


# -------------------------------------------------------------------------
# P-Value Computation
# -------------------------------------------------------------------------
def compute_mannwhitney_pvalue(group1: pd.Series, group2: pd.Series) -> Optional[float]:
    """Compute Mann–Whitney U test (non-parametric test for median differences)."""
    try:
        g1, g2 = group1.dropna(), group2.dropna()
        if len(g1) == 0 or len(g2) == 0:
            return np.nan
        _, p_value = mannwhitneyu(g1, g2, alternative="two-sided")
        return float(p_value)
    except Exception as e:
        logger.warning(f"Mann–Whitney test failed: {e}")
        return np.nan


def compute_ks_pvalue(group1: pd.Series, group2: pd.Series) -> Optional[float]:
    """Compute Kolmogorov–Smirnov test (distribution shape difference)."""
    try:
        g1, g2 = group1.dropna(), group2.dropna()
        if len(g1) == 0 or len(g2) == 0:
            return np.nan
        _, p_value = ks_2samp(g1, g2)
        return float(p_value)
    except Exception as e:
        logger.warning(f"KS test failed: {e}")
        return np.nan


def compute_ttest_pvalue(
    group1: pd.Series,
    group2: pd.Series,
    equal_var: bool = False,
) -> Optional[float]:
    """Compute Welch’s t-test (parametric mean comparison)."""
    try:
        g1, g2 = group1.dropna(), group2.dropna()
        if len(g1) == 0 or len(g2) == 0:
            return np.nan
        _, p_value = ttest_ind(g1, g2, equal_var=equal_var)
        return float(p_value)
    except Exception as e:
        logger.warning(f"t-test failed: {e}")
        return np.nan


# -------------------------------------------------------------------------
# Multiple Hypothesis Correction
# -------------------------------------------------------------------------
def adjust_pvalues(p_values: List[float], method: str = "fdr_bh") -> np.ndarray:
    """Apply multiple testing correction (e.g. FDR, Bonferroni).

    Args:
        p_values (List[float]): List or array of raw p-values.
        method (str, optional): Correction method. Defaults to 'fdr_bh'.

    Returns:
        np.ndarray: Array of adjusted p-values.
    """
    try:
        _, adj_pvals, _, _ = multipletests(p_values, method=method)
        return adj_pvals
    except Exception as e:
        logger.warning(f"Multiple testing correction failed: {e}")
        return np.array(p_values)


# -------------------------------------------------------------------------
# Feature Separation Scoring
# -------------------------------------------------------------------------
def compute_feature_separation(
    df: pd.DataFrame,
    features: List[str],
    target: str = "signal",
    method: Literal["mannwhitney", "ks", "ttest"] = "mannwhitney",
) -> pd.DataFrame:
    """Compute per-feature p-values for signal/background separation.

    Args:
        df (pd.DataFrame): Input dataset with features and target.
        features (List[str]): Feature column names to test.
        target (str, optional): Binary target column name. Defaults to "signal".
        method (Literal["mannwhitney", "ks", "ttest"], optional): Statistical test type.

    Returns:
        pd.DataFrame: DataFrame with columns ['feature', 'p_value', 'method'].
    """
    df = df.copy()
    results = []
    for feature in features:
        try:
            g1 = df[df[target] == 0][feature]
            g2 = df[df[target] == 1][feature]
            if method == "mannwhitney":
                pval = compute_mannwhitney_pvalue(g1, g2)
            elif method == "ks":
                pval = compute_ks_pvalue(g1, g2)
            elif method == "ttest":
                pval = compute_ttest_pvalue(g1, g2)
            else:
                raise ValueError(f"Unknown method: {method}")
            results.append((feature, pval, method))
        except Exception as e:
            logger.warning(f"Failed to compute separation for '{feature}': {e}")
            results.append((feature, np.nan, method))
    return pd.DataFrame(results, columns=["feature", "p_value", "method"])


# -------------------------------------------------------------------------
# AUC (Discriminative Power)
# -------------------------------------------------------------------------
def compute_auc_score(df: pd.DataFrame, feature: str, target: str = "signal") -> Optional[float]:
    """Compute Area Under ROC Curve (AUC) for a feature’s discriminative power.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature (str): Feature column name.
        target (str, optional): Binary target column. Defaults to "signal".

    Returns:
        Optional[float]: AUC value (0.5 = no discrimination, 1.0 = perfect).
    """
    try:
        x, y = df[feature].values, df[target].values
        if len(np.unique(y)) != 2:
            logger.warning("Target must be binary for AUC computation.")
            return np.nan
        return roc_auc_score(y, x)
    except Exception as e:
        logger.warning(f"Failed to compute AUC for '{feature}': {e}")
        return np.nan


# -------------------------------------------------------------------------
# Jensen–Shannon Divergence
# -------------------------------------------------------------------------
def compute_js_divergence(
    group1: np.ndarray,
    group2: np.ndarray,
    bins: int = 50,
) -> float:
    """Compute Jensen–Shannon divergence (distribution similarity measure).

    Args:
        group1 (np.ndarray): First distribution sample.
        group2 (np.ndarray): Second distribution sample.
        bins (int, optional): Number of histogram bins. Defaults to 50.

    Returns:
        float: JS divergence in range [0, 1] (0 = identical, 1 = disjoint).
    """
    p_hist, _ = np.histogram(group1, bins=bins, density=True)
    q_hist, _ = np.histogram(group2, bins=bins, density=True)
    p_hist, q_hist = np.clip(p_hist, 1e-12, None), np.clip(q_hist, 1e-12, None)
    m = 0.5 * (p_hist + q_hist)
    return 0.5 * (entropy(p_hist, m) + entropy(q_hist, m))


# -------------------------------------------------------------------------
# Redundancy Analysis
# -------------------------------------------------------------------------
def compute_redundancy_matrix(df: pd.DataFrame, threshold: float = 0.9) -> list[tuple[str, str]]:
    """Identify pairs of features with high correlation (redundancy).

    Args:
        df (pd.DataFrame): Numeric DataFrame.
        threshold (float, optional): Correlation threshold. Defaults to 0.9.

    Returns:
        list[tuple[str, str]]: List of redundant feature pairs.
    """
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    redundant = [
        (col, row)
        for col in upper.columns
        for row in upper.index
        if upper.loc[row, col] > threshold
    ]
    return redundant
