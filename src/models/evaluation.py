"""
Statistical evaluation utilities — uncertainty and significance around model metrics.

Point-estimate metrics (a single AUC number) don't say whether a difference
between two models is real or noise. This module adds the three checks that
answer that:

    bootstrap_metric_ci   — confidence interval around a single metric
    mcnemar_test          — is model A significantly better than model B on
                             the *same* test set (paired comparison)?
    permutation_importance_report — which features matter, with a
                             significance flag instead of a bare ranking

Usage
-----
    from src.models.evaluation import bootstrap_metric_ci, mcnemar_test

    ci = bootstrap_metric_ci(y_test, y_proba, roc_auc_score)
    cmp = mcnemar_test(y_test, y_pred_a, y_pred_b)
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import binomtest
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.inspection import permutation_importance


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> Dict:
    """
    Bootstrap confidence interval for a metric (e.g. roc_auc_score, f1_score).

    Resamples (y_true, y_score) with replacement `n_boot` times, recomputes
    the metric on each resample, and takes the percentile interval. Resamples
    that end up single-class (undefined AUC) are skipped rather than crashing
    the whole run — this happens more often on small or imbalanced test sets.

    Returns
    -------
    dict with `point`, `lower`, `upper`, `ci`, and `n_boot` (resamples actually used)
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    point = float(metric_fn(y_true, y_score))

    boot_stats: List[float] = []
    with warnings.catch_warnings():
        # a resample can land on a single class (undefined AUC) — expected
        # and handled below via the NaN check, not something to log per-resample
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            try:
                stat = metric_fn(y_true[idx], y_score[idx])
            except ValueError:
                continue  # e.g. only one class present in this resample
            # some sklearn metrics (e.g. roc_auc_score) warn and return NaN for
            # a degenerate resample instead of raising — treat that the same way
            if stat is None or (isinstance(stat, float) and np.isnan(stat)):
                continue
            boot_stats.append(stat)

    if len(boot_stats) < n_boot * 0.5:
        logger.warning(
            f"bootstrap_metric_ci: only {len(boot_stats)}/{n_boot} resamples were "
            "usable — CI may be unstable. Check class balance in y_true."
        )

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1 - alpha) * 100))

    return {
        "point": round(point, 4),
        "lower": round(lower, 4),
        "upper": round(upper, 4),
        "ci": ci,
        "n_boot": len(boot_stats),
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> Dict:
    """
    Exact McNemar's test — is model A significantly different from model B
    on the *same* test set?

    Compares only the samples where the two models disagree (discordant
    pairs): `b` = A right & B wrong, `c` = A wrong & B right. Under the null
    hypothesis (no real difference), b and c should each be ~half of the
    discordant total — tested with an exact binomial test rather than the
    chi-square approximation, which is unreliable when b + c is small.

    Returns
    -------
    dict with `b`, `c`, `n_discordant`, `p_value`, and `favors` ("a"/"b"/"tie")
    """
    y_true = np.asarray(y_true)
    correct_a = np.asarray(y_pred_a) == y_true
    correct_b = np.asarray(y_pred_b) == y_true

    b = int(np.sum(correct_a & ~correct_b))   # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))   # A wrong, B right
    n = b + c

    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p_value": 1.0,
                "favors": "tie", "significant": False}

    p_value = binomtest(min(b, c), n, 0.5).pvalue
    favors = "tie" if b == c else ("a" if b > c else "b")

    return {
        "b": b,
        "c": c,
        "n_discordant": n,
        "p_value": round(float(p_value), 4),
        "favors": favors,
        "significant": bool(p_value < 0.05),
    }


def permutation_importance_report(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    random_state: int = 42,
    scoring: str = "roc_auc",
) -> pd.DataFrame:
    """
    Permutation feature importance with a significance flag, instead of a
    bare ranking of `model.feature_importances_`.

    Shuffles each feature `n_repeats` times and measures the drop in
    `scoring`; a feature is flagged significant when its importance is more
    than 2 standard deviations above zero across repeats (i.e. the drop from
    shuffling it isn't explainable by repeat-to-repeat noise alone).

    Returns
    -------
    pd.DataFrame sorted by importance_mean descending, with `significant` column
    """
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats,
        random_state=random_state, scoring=scoring, n_jobs=-1,
    )

    df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })
    df["significant"] = df["importance_mean"] > 2 * df["importance_std"]
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
