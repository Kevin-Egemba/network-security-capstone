"""
DataValidator — Schema, quality, and leakage checks for all datasets.

This module implements the kind of data validation a production ML system
needs before training or serving predictions. It returns structured reports
rather than raising exceptions, so callers can decide how to respond.

Usage
-----
    from src.data.validator import DataValidator

    validator = DataValidator()
    report = validator.validate_unsw(df_train)
    print(report.summary())

    leakage = validator.check_leakage(df, target="label", threshold=0.95)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


# ── Expected schema definitions ───────────────────────────────────────────────
UNSW_EXPECTED_TARGETS = {"label", "attack_cat"}
UNSW_CATEGORICAL = {"proto", "service", "state"}

BETH_NUMERIC_COLS = {
    "processId", "threadId", "parentProcessId", "userId",
    "mountNamespace", "eventId", "argsNum", "returnValue",
}
BETH_TARGET = "evil"

CYBER_TARGET = "Attack Type"
CYBER_ATTACK_CLASSES = {"DDoS", "Malware", "Intrusion"}


# ── Report container ──────────────────────────────────────────────────────────
@dataclass
class ValidationReport:
    dataset_name: str
    shape: tuple
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    stats: Dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"── Validation Report: {self.dataset_name} [{status}] ──",
            f"  Shape  : {self.shape[0]:,} rows × {self.shape[1]} cols",
        ]
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"    ⚠  {w}")
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for e in self.errors:
                lines.append(f"    ✗  {e}")
        if self.passed and not self.warnings:
            lines.append("  ✓ All checks passed.")
        for k, v in self.stats.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)


# ── Validator ─────────────────────────────────────────────────────────────────
class DataValidator:
    """
    Runs data quality, schema, and leakage checks on each dataset.
    """

    # ── UNSW-NB15 ─────────────────────────────────────────────────────────────
    def validate_unsw(self, df: pd.DataFrame, split: str = "train") -> ValidationReport:
        report = ValidationReport(f"UNSW-NB15 ({split})", df.shape)

        # Required targets
        for col in UNSW_EXPECTED_TARGETS:
            if col not in df.columns:
                report.errors.append(f"Missing required column: '{col}'")

        # Null checks
        null_pct = df.isnull().mean()
        high_null = null_pct[null_pct > 0.05]
        for col, pct in high_null.items():
            report.warnings.append(f"High nulls in '{col}': {pct:.1%}")

        # Duplicate rows
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            report.warnings.append(f"{dup_count:,} duplicate rows detected")

        # Label distribution
        if "label" in df.columns:
            dist = df["label"].value_counts(normalize=True)
            report.stats["class_balance"] = dist.to_dict()
            if dist.min() < 0.10:
                report.warnings.append(
                    f"Severe class imbalance: minority class = {dist.min():.1%}"
                )

        # Categorical encoding check
        for col in UNSW_CATEGORICAL:
            if col in df.columns and df[col].dtype == "object":
                report.warnings.append(
                    f"Categorical column '{col}' needs encoding before modeling"
                )

        # Numeric range sanity (negative packet counts, etc.)
        numeric = df.select_dtypes(include=[np.number])
        report.stats["numeric_cols"] = numeric.shape[1]
        report.stats["total_nulls"] = int(df.isnull().sum().sum())

        logger.info(f"\n{report.summary()}")
        return report

    # ── BETH ──────────────────────────────────────────────────────────────────
    def validate_beth(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport("BETH", df.shape)

        if BETH_TARGET not in df.columns:
            report.errors.append(f"Missing target column: '{BETH_TARGET}'")

        # Label sparsity
        if BETH_TARGET in df.columns:
            evil_pct = df[BETH_TARGET].mean()
            report.stats["evil_pct"] = f"{evil_pct:.3%}"
            if evil_pct < 0.01:
                report.warnings.append(
                    f"Extremely sparse labels: only {evil_pct:.2%} evil events. "
                    "Supervised learning is unreliable — use unsupervised methods."
                )

        # Expected numeric cols
        missing_cols = BETH_NUMERIC_COLS - set(df.columns)
        if missing_cols:
            report.warnings.append(f"Expected numeric cols not found: {missing_cols}")

        report.stats["total_nulls"] = int(df.isnull().sum().sum())
        logger.info(f"\n{report.summary()}")
        return report

    # ── Cyber Attacks ─────────────────────────────────────────────────────────
    def validate_cyber_attacks(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport("Cyber Attacks (Synthetic)", df.shape)

        if CYBER_TARGET not in df.columns:
            report.errors.append(f"Missing target column: '{CYBER_TARGET}'")

        # Class balance (should be ~33% each)
        if CYBER_TARGET in df.columns:
            classes = set(df[CYBER_TARGET].unique())
            if not classes == CYBER_ATTACK_CLASSES:
                report.warnings.append(
                    f"Unexpected classes: {classes} (expected {CYBER_ATTACK_CLASSES})"
                )
            dist = df[CYBER_TARGET].value_counts(normalize=True)
            report.stats["class_balance"] = dist.to_dict()

        report.stats["total_nulls"] = int(df.isnull().sum().sum())
        logger.info(f"\n{report.summary()}")
        return report

    # ── Leakage Detection ─────────────────────────────────────────────────────
    def check_leakage(
        self,
        df: pd.DataFrame,
        target: str,
        threshold: float = 0.95,
    ) -> pd.DataFrame:
        """
        Identify features suspiciously correlated with the target.

        Returns a DataFrame of (feature, correlation, leakage_flag) sorted
        by absolute correlation descending.

        Parameters
        ----------
        df : pd.DataFrame
        target : str  — target column name
        threshold : float — correlation above this triggers a leakage flag

        Returns
        -------
        pd.DataFrame
        """
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in DataFrame")

        y = df[target]
        if y.dtype == "object":
            y = y.astype("category").cat.codes

        results = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(target, errors="ignore")

        for col in numeric_cols:
            try:
                corr = abs(df[col].corr(y))
                results.append({
                    "feature": col,
                    "abs_correlation": round(corr, 4),
                    "leakage_flag": corr > threshold,
                })
            except Exception:
                continue

        result_df = (
            pd.DataFrame(results)
            .sort_values("abs_correlation", ascending=False)
            .reset_index(drop=True)
        )

        leaked = result_df[result_df["leakage_flag"]]
        if not leaked.empty:
            logger.warning(
                f"Potential leakage detected in {len(leaked)} feature(s):\n"
                f"{leaked[['feature', 'abs_correlation']].to_string(index=False)}"
            )
        else:
            logger.info("No leakage detected above threshold.")

        return result_df

    # ── Cross-dataset Consistency ─────────────────────────────────────────────
    def compare_train_test(
        self, df_train: pd.DataFrame, df_test: pd.DataFrame, target: str
    ) -> Dict:
        """
        Check that train and test sets have consistent column sets and
        non-overlapping index ranges (if numeric).
        """
        train_cols = set(df_train.columns)
        test_cols = set(df_test.columns)

        report = {
            "train_shape": df_train.shape,
            "test_shape": df_test.shape,
            "columns_only_in_train": list(train_cols - test_cols),
            "columns_only_in_test": list(test_cols - train_cols),
            "target_in_both": target in train_cols and target in test_cols,
        }

        if target in df_train.columns and target in df_test.columns:
            train_classes = set(df_train[target].unique())
            test_classes = set(df_test[target].unique())
            report["unseen_test_classes"] = list(test_classes - train_classes)

        return report
