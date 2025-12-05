"""
Unit tests for the statistical_analysis module.

Covers hypothesis tests, effect sizes, bootstrap confidence intervals,
AUC computation, and redundancy detection using MiniBooNE-like synthetic data.
"""

import pytest
import numpy as np
import pandas as pd
from src.stats import statistical_analysis as sa


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


@pytest.fixture
def sample_groups():
    """Provide two synthetic groups representing signal and background."""
    np.random.seed(42)
    group1 = pd.Series(np.random.normal(0, 1, 200))  # background
    group2 = pd.Series(np.random.normal(1, 1, 200))  # signal (shifted mean)
    return group1, group2


@pytest.fixture
def df_synthetic():
    """Synthetic DataFrame similar to MiniBooNE-style data."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "feature_1": np.random.normal(0, 1, 300),
            "feature_2": np.random.normal(2, 1.5, 300),
            "feature_3": np.random.exponential(1, 300),
            "signal": np.random.choice([0, 1], 300, p=[0.6, 0.4]),
        }
    )
    return df


# -------------------------------------------------------------------------
# Bootstrapping Tests
# -------------------------------------------------------------------------


def test_compute_bootstrap_error_valid():
    """Test bootstrap error estimation for binary column."""
    s = pd.Series(np.random.choice(["A", "B"], size=200))
    err = sa.compute_bootstrap_error(s, label_value="A", n_boot=100)
    assert isinstance(err, float)
    assert err >= 0


def test_compute_bootstrap_error_empty():
    """Should return NaN for empty series."""
    s = pd.Series(dtype=str)
    assert np.isnan(sa.compute_bootstrap_error(s, "A"))


def test_bootstrap_ci_mean_and_median(sample_groups):
    """Test bootstrap confidence interval computation."""
    group1, _ = sample_groups
    ci_mean = sa.bootstrap_ci(group1, stat="mean", n_boot=100)
    ci_median = sa.bootstrap_ci(group1, stat="median", n_boot=100)
    assert len(ci_mean) == 2
    assert len(ci_median) == 2
    assert ci_mean[0] < ci_mean[1]


# -------------------------------------------------------------------------
# Effect Size Metrics
# -------------------------------------------------------------------------
def test_compute_effect_size_cohens_d(sample_groups):
    """Cohen's d should indicate a large magnitude difference between groups."""
    g1, g2 = sample_groups
    effect = sa.compute_effect_size(g1, g2, method="cohen_d")
    assert isinstance(effect, float)
    assert 0.5 < abs(effect) < 2.5


def test_compute_effect_size_cliffs_delta(sample_groups):
    """Test Cliff’s delta effect size."""
    g1, g2 = sample_groups
    delta = sa.compute_effect_size(g1, g2, method="cliffs_delta")
    assert -1 <= delta <= 1


# -------------------------------------------------------------------------
# Hypothesis Testing
# -------------------------------------------------------------------------


def test_mannwhitney_pvalue_significant(sample_groups):
    """Mann–Whitney test should detect difference between groups."""
    g1, g2 = sample_groups
    p = sa.compute_mannwhitney_pvalue(g1, g2)
    assert p < 0.05


def test_ks_pvalue_significant(sample_groups):
    """KS test should detect distribution difference."""
    g1, g2 = sample_groups
    p = sa.compute_ks_pvalue(g1, g2)
    assert p < 0.05


def test_ttest_pvalue_significant(sample_groups):
    """Welch’s t-test should detect difference."""
    g1, g2 = sample_groups
    p = sa.compute_ttest_pvalue(g1, g2)
    assert p < 0.05


def test_adjust_pvalues_fdr():
    """Test multiple hypothesis correction."""
    raw = np.array([0.001, 0.02, 0.5, 0.04])
    adjusted = sa.adjust_pvalues(raw, method="fdr_bh")
    assert all(adjusted >= raw)
    assert len(adjusted) == len(raw)


# -------------------------------------------------------------------------
# Feature Separation (Batch)
# -------------------------------------------------------------------------


def test_compute_feature_separation(df_synthetic):
    """Ensure p-values computed for multiple features."""
    results = sa.compute_feature_separation(
        df=df_synthetic,
        features=["feature_1", "feature_2"],
        target="signal",
        method="mannwhitney",
    )
    assert set(results.columns) == {"feature", "p_value", "method"}
    assert len(results) == 2
    assert all(results["p_value"].between(0, 1, inclusive="both"))


# -------------------------------------------------------------------------
# AUC Computation
# -------------------------------------------------------------------------


def test_compute_auc_score(df_synthetic):
    """Test feature AUC computation."""
    auc = sa.compute_auc_score(df_synthetic, feature="feature_1", target="signal")
    assert 0.0 <= auc <= 1.0


def test_compute_auc_score_invalid_target(df_synthetic):
    """Should return NaN if target not binary."""
    df_synthetic["signal"] = 2  # make target constant
    auc = sa.compute_auc_score(df_synthetic, "feature_1", target="signal")
    assert np.isnan(auc)


# -------------------------------------------------------------------------
# Jensen–Shannon Divergence
# -------------------------------------------------------------------------


def test_compute_js_divergence(sample_groups):
    """JS divergence should be small for similar distributions."""
    g1, g2 = sample_groups
    jsd = sa.compute_js_divergence(g1.values, g2.values)
    assert 0 <= jsd <= 1


def test_compute_js_divergence_identical_distribution():
    """JS divergence ≈ 0 for identical samples."""
    x = np.random.normal(0, 1, 200)
    jsd = sa.compute_js_divergence(x, x)
    assert jsd < 1e-5


# -------------------------------------------------------------------------
# Redundancy Analysis
# -------------------------------------------------------------------------


def test_compute_redundancy_matrix(df_synthetic):
    """Should return redundant feature pairs for high correlation."""
    df = df_synthetic.copy()
    df["feature_4"] = df["feature_1"] * 1.01  # correlated with feature_1
    redundant = sa.compute_redundancy_matrix(df, threshold=0.9)
    assert any("feature_1" in pair for pair in redundant)
