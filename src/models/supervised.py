"""
Supervised classification models for network intrusion detection.

TwoStageDetector
    Stage 1: Binary detection (normal vs attack) — optimised for recall
    Stage 2: Multi-class classification (attack type) — among flagged traffic

AttackClassifier
    Standalone multi-class classifier with SMOTE and experiment tracking.

Both classes:
  - Wrap sklearn-compatible estimators
  - Log metrics with MLflow if available
  - Serialize/deserialize with joblib
  - Return structured result dicts suitable for DB storage

Usage
-----
    from src.models.supervised import TwoStageDetector
    from src.data.preprocessor import UNSWPreprocessor

    prep = UNSWPreprocessor()
    X_train, y_train = prep.fit_transform(df_train, target="label")
    X_test, y_test   = prep.transform(df_test, target="label")

    detector = TwoStageDetector()
    results = detector.fit(X_train, y_train)
    eval_results = detector.evaluate(X_test, y_test)
    detector.save("models/two_stage_detector.pkl")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("XGBoost not installed — falling back to HistGradientBoosting")

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False

from src.config import settings


def _make_xgb(**kwargs) -> Any:
    if _HAS_XGB:
        return XGBClassifier(
            eval_metric="logloss", use_label_encoder=False,
            random_state=settings.random_seed, **kwargs
        )
    return HistGradientBoostingClassifier(random_state=settings.random_seed)


# ─────────────────────────────────────────────────────────────────────────────
# Two-Stage Detector (mirrors real SOC IDS workflow)
# ─────────────────────────────────────────────────────────────────────────────
class TwoStageDetector:
    """
    Stage 1: Binary classifier — flag suspicious traffic (high recall).
    Stage 2: Multi-class classifier — categorise attack type among flagged.

    Parameters
    ----------
    stage1_model : str
        'rf' | 'xgb' | 'lr'
    stage2_model : str
        'rf' | 'xgb'
    """

    def __init__(self, stage1_model: str = "xgb", stage2_model: str = "rf"):
        self.stage1_model = self._build_stage1(stage1_model)
        self.stage2_model = self._build_stage2(stage2_model)
        self.stage1_name = stage1_model
        self.stage2_name = stage2_model
        self._is_fit = False

    # ── Build estimators ──────────────────────────────────────────────────────
    def _build_stage1(self, name: str) -> Any:
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=200, max_depth=15, class_weight="balanced",
                n_jobs=settings.n_jobs, random_state=settings.random_seed
            )
        if name == "lr":
            return LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced",
                random_state=settings.random_seed
            )
        return _make_xgb(n_estimators=300, max_depth=6, learning_rate=0.1,
                         scale_pos_weight=2)

    def _build_stage2(self, name: str) -> Any:
        if name == "xgb":
            return _make_xgb(n_estimators=300, max_depth=6, learning_rate=0.1)
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=settings.n_jobs, random_state=settings.random_seed
        )

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray,
        y_binary: np.ndarray,
        X_train_multi: Optional[np.ndarray] = None,
        y_multi: Optional[np.ndarray] = None,
        cv: bool = True,
    ) -> Dict:
        """
        Fit both stages.

        If X_train_multi / y_multi are None, Stage 2 is trained on the
        attack-only subset of X_train / y_binary (y_binary used as filter).
        """
        logger.info("Fitting Stage 1: binary detection…")
        self.stage1_model.fit(X_train, y_binary)

        # Stage 2 — train on attack samples only
        if X_train_multi is not None and y_multi is not None:
            X2, y2 = X_train_multi, y_multi
        else:
            attack_mask = y_binary == 1
            X2, y2 = X_train[attack_mask], y_binary[attack_mask]

        if _HAS_SMOTE and len(np.unique(y2)) > 1:
            try:
                smote = SMOTE(random_state=settings.random_seed)
                X2, y2 = smote.fit_resample(X2, y2)
                logger.info(f"SMOTE applied → {len(y2):,} attack samples")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}")

        logger.info("Fitting Stage 2: attack classification…")
        self.stage2_model.fit(X2, y2)
        self._is_fit = True

        results = {"stage": "fit", "stage1_model": self.stage1_name,
                   "stage2_model": self.stage2_name}

        if cv:
            cv_scores = cross_val_score(
                self.stage1_model, X_train, y_binary,
                cv=3, scoring="roc_auc", n_jobs=settings.n_jobs
            )
            results["stage1_cv_auc"] = cv_scores.mean().round(4)
            logger.info(f"Stage 1 CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return results

    # ── Prediction ────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run both stages. Returns binary predictions and attack-type predictions
        for flagged samples.
        """
        if not self._is_fit:
            raise RuntimeError("Model not fit yet. Call .fit() first.")
        y_binary = self.stage1_model.predict(X)
        y_proba = self.stage1_model.predict_proba(X)[:, 1]

        attack_mask = y_binary == 1
        y_attack_type = np.full(len(X), "normal", dtype=object)
        if attack_mask.sum() > 0:
            y_attack_type[attack_mask] = self.stage2_model.predict(X[attack_mask])

        return {
            "binary_pred": y_binary,
            "attack_proba": y_proba,
            "attack_type": y_attack_type,
            "n_flagged": int(attack_mask.sum()),
        }

    # ── Evaluation ────────────────────────────────────────────────────────────
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        preds = self.predict(X_test)
        y_pred = preds["binary_pred"]
        y_proba = preds["attack_proba"]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred),
        }

        logger.info(
            f"Stage 1 Evaluation — Accuracy: {metrics['accuracy']:.4f}  "
            f"AUC: {metrics['roc_auc']:.4f}  F1: {metrics['f1_weighted']:.4f}"
        )

        if _HAS_MLFLOW:
            self._log_mlflow(metrics)

        return metrics

    def _log_mlflow(self, metrics: Dict) -> None:
        try:
            with mlflow.start_run(run_name="two_stage_detector"):
                mlflow.log_params({
                    "stage1_model": self.stage1_name,
                    "stage2_model": self.stage2_name,
                })
                mlflow.log_metrics({k: v for k, v in metrics.items()
                                    if isinstance(v, (int, float))})
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")

    # ── Serialization ─────────────────────────────────────────────────────────
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"TwoStageDetector saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TwoStageDetector":
        return joblib.load(path)

    def feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Return Stage 1 feature importances as a sorted DataFrame."""
        if not hasattr(self.stage1_model, "feature_importances_"):
            raise AttributeError("Stage 1 model does not have feature_importances_")
        imp = self.stage1_model.feature_importances_
        return (
            pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Attack Classifier
# ─────────────────────────────────────────────────────────────────────────────
class AttackClassifier:
    """General multi-class classifier comparing multiple algorithms."""

    MODELS = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=settings.random_seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=settings.n_jobs, random_state=settings.random_seed
        ),
        "xgboost": _make_xgb(n_estimators=300, max_depth=6, learning_rate=0.1),
    }

    def __init__(self, model_name: str = "random_forest"):
        if model_name not in self.MODELS:
            raise ValueError(f"model_name must be one of {list(self.MODELS)}")
        self.model_name = model_name
        self.model = self.MODELS[model_name]
        self._is_fit = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AttackClassifier":
        self.model.fit(X, y)
        self._is_fit = True
        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray, class_names: List[str] = None) -> Dict:
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)

        metrics = {
            "model": self.model_name,
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "f1_macro": round(f1_score(y, y_pred, average="macro"), 4),
            "f1_weighted": round(f1_score(y, y_pred, average="weighted"), 4),
            "classification_report": classification_report(
                y, y_pred, target_names=class_names
            ),
        }

        try:
            metrics["roc_auc_ovr"] = round(
                roc_auc_score(y, y_proba, multi_class="ovr"), 4
            )
        except Exception:
            pass

        logger.info(
            f"{self.model_name}: acc={metrics['accuracy']:.4f}  "
            f"f1={metrics['f1_weighted']:.4f}"
        )
        return metrics

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "AttackClassifier":
        return joblib.load(path)
