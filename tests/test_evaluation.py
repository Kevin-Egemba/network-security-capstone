"""Tests for src.models.evaluation — bootstrap CI, McNemar's test, permutation importance."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

from src.models.evaluation import (
    bootstrap_metric_ci,
    mcnemar_test,
    permutation_importance_report,
)


@pytest.fixture
def separable_binary():
    """A binary problem a classifier can solve well (not perfectly)."""
    rng = np.random.default_rng(42)
    n = 300
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    # add label noise so AUC is high but not 1.0, and bootstrap has spread
    flip = rng.choice(n, size=15, replace=False)
    y[flip] = 1 - y[flip]
    return X, y


class TestBootstrapMetricCI:
    def test_ci_contains_point_estimate(self, separable_binary):
        X, y = separable_binary
        model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X, y)
        y_proba = model.predict_proba(X)[:, 1]

        result = bootstrap_metric_ci(y, y_proba, roc_auc_score, n_boot=200)

        assert result["lower"] <= result["point"] <= result["upper"]
        assert 0.0 <= result["lower"] <= 1.0
        assert 0.0 <= result["upper"] <= 1.0
        assert result["n_boot"] > 100  # most resamples should be usable

    def test_ci_narrows_with_more_data(self):
        """Bootstrap CI width should shrink as sample size grows (basic sanity check)."""
        rng = np.random.default_rng(0)

        def make(n):
            y_true = rng.integers(0, 2, n)
            y_score = y_true + rng.normal(0, 0.5, n)
            return y_true, y_score

        small = bootstrap_metric_ci(*make(30), roc_auc_score, n_boot=300, random_state=1)
        large = bootstrap_metric_ci(*make(300), roc_auc_score, n_boot=300, random_state=1)

        assert (large["upper"] - large["lower"]) < (small["upper"] - small["lower"])

    def test_skips_degenerate_resamples_without_crashing(self):
        """A tiny, heavily imbalanced set should still return a result, not raise."""
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        y_score = np.array([0.1, 0.2, 0.15, 0.3, 0.05, 0.4, 0.25, 0.35, 0.1, 0.9])

        result = bootstrap_metric_ci(y_true, y_score, roc_auc_score, n_boot=200)
        assert result["n_boot"] > 0


class TestMcNemarTest:
    def test_identical_predictions_are_a_tie(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])  # same for both models

        result = mcnemar_test(y_true, y_pred, y_pred.copy())
        assert result["n_discordant"] == 0
        assert result["p_value"] == 1.0
        assert result["favors"] == "tie"

    def test_detects_a_clear_difference(self):
        """Model A wrong on none of the discordant cases, B wrong on all → significant."""
        y_true = np.ones(40, dtype=int)
        y_pred_a = np.ones(40, dtype=int)          # always correct
        y_pred_b = np.zeros(40, dtype=int)          # always wrong

        result = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert result["n_discordant"] == 40
        assert result["favors"] == "a"
        assert result["significant"] is True
        assert result["p_value"] < 0.001

    def test_no_disagreement_between_models_is_not_significant(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, 60)
        # both models wrong on exactly the same handful of cases → 0 discordant
        y_pred = y_true.copy()
        y_pred[:5] = 1 - y_pred[:5]

        result = mcnemar_test(y_true, y_pred, y_pred.copy())
        assert result["significant"] is False


class TestPermutationImportanceReport:
    def test_informative_feature_ranks_above_noise_feature(self, separable_binary):
        X, y = separable_binary
        # add a pure-noise feature that shouldn't matter
        rng = np.random.default_rng(1)
        X_with_noise = np.hstack([X, rng.normal(size=(len(X), 1))])
        feature_names = ["f0", "f1", "f2", "f3", "noise"]

        model = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_with_noise, y)
        report = permutation_importance_report(
            model, X_with_noise, y, feature_names, n_repeats=15, random_state=42
        )

        assert list(report.columns) == ["feature", "importance_mean", "importance_std", "significant"]
        assert report.iloc[0]["feature"] in {"f0", "f1"}  # the actually-informative features
        noise_row = report[report["feature"] == "noise"].iloc[0]
        assert noise_row["importance_mean"] < report.iloc[0]["importance_mean"]
