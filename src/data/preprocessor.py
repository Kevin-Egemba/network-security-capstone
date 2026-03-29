"""
Feature engineering pipelines for each dataset.

Each preprocessor wraps scikit-learn pipelines so they can be:
  - fit on train, applied to test (no leakage)
  - serialized with joblib for serving
  - inspected for feature names

Usage
-----
    from src.data.preprocessor import UNSWPreprocessor

    prep = UNSWPreprocessor()
    X_train, y_train = prep.fit_transform(df_train, target="label")
    X_test, y_test  = prep.transform(df_test, target="label")
    prep.save("models/unsw_preprocessor.pkl")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)


# ─────────────────────────────────────────────────────────────────────────────
# UNSW-NB15 Preprocessor
# ─────────────────────────────────────────────────────────────────────────────
class UNSWPreprocessor:
    """
    Two-stage pipeline for UNSW-NB15:
      1. Separate binary target (label) and multiclass target (attack_cat)
      2. Standard-scale numerics, one-hot encode categoricals
      3. Return (X, y) arrays ready for sklearn estimators
    """

    CATEGORICAL_COLS = ["proto", "service", "state"]
    DROP_COLS = ["id"]
    TARGET_BINARY = "label"
    TARGET_MULTI = "attack_cat"

    def __init__(self, scaler: str = "standard"):
        scaler_cls = StandardScaler if scaler == "standard" else RobustScaler
        self._pipeline: Optional[Pipeline] = None
        self._scaler_cls = scaler_cls
        self._feature_names: Optional[List[str]] = None
        self._label_encoder = LabelEncoder()

    def _build_pipeline(self, numeric_cols: List[str]) -> Pipeline:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", self._scaler_cls()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_cols),
                ("cat", categorical_transformer, self.CATEGORICAL_COLS),
            ],
            remainder="drop",
        )
        return Pipeline([("preprocessor", preprocessor)])

    def _prepare(
        self, df: pd.DataFrame, target: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower()

        # Normalise target name
        target = target.lower()
        if target not in df.columns:
            raise KeyError(f"Target '{target}' not in DataFrame columns")

        y = df.pop(target)

        # Drop the other target and id columns
        for col in self.DROP_COLS + [c for c in [self.TARGET_BINARY, self.TARGET_MULTI]
                                      if c != target and c in df.columns]:
            df.drop(columns=[col], inplace=True, errors="ignore")

        cat_cols = [c for c in self.CATEGORICAL_COLS if c in df.columns]
        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in cat_cols]
        self._num_cols = num_cols
        self._cat_cols = cat_cols
        return df, y

    def fit_transform(
        self, df: pd.DataFrame, target: str = "label"
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_df, y = self._prepare(df, target)
        self._pipeline = self._build_pipeline(self._num_cols)
        X = self._pipeline.fit_transform(X_df)

        if y.dtype == object:
            y_enc = self._label_encoder.fit_transform(y)
        else:
            y_enc = y.values
            self._label_encoder = None  # binary — no encoding needed

        self._feature_names = self._build_feature_names()
        logger.info(f"UNSWPreprocessor fit: {X.shape[1]} features, {len(y_enc)} samples")
        return X, y_enc

    def transform(
        self, df: pd.DataFrame, target: str = "label"
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._pipeline is None:
            raise RuntimeError("Call fit_transform before transform")
        X_df, y = self._prepare(df, target)
        X = self._pipeline.transform(X_df)

        if self._label_encoder is not None:
            y_enc = self._label_encoder.transform(y)
        else:
            y_enc = y.values
        return X, y_enc

    def _build_feature_names(self) -> List[str]:
        names = list(self._num_cols)
        try:
            ohe = self._pipeline.named_steps["preprocessor"].named_transformers_["cat"]
            cat_names = ohe.named_steps["onehot"].get_feature_names_out(self._cat_cols)
            names += list(cat_names)
        except Exception:
            pass
        return names

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names or []

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "UNSWPreprocessor":
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# BETH Preprocessor
# ─────────────────────────────────────────────────────────────────────────────
class BETHPreprocessor:
    """
    Feature engineering for BETH system-call telemetry.

    Drops high-cardinality identifier columns (processId etc.) and scales
    the remaining numeric behavioural features.
    """

    ID_COLUMNS = [
        "processId", "threadId", "parentProcessId", "userId",
        "mountNamespace", "returnValue", "argsNum",
        "processName", "hostName",
    ]
    TARGET_COL = "evil"

    def __init__(self):
        self._pipeline: Optional[Pipeline] = None
        self._feature_cols: Optional[List[str]] = None

    def _get_feature_cols(self, df: pd.DataFrame) -> List[str]:
        drop = set(self.ID_COLUMNS + [self.TARGET_COL])
        return [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in drop]

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        df = df.copy()
        self._feature_cols = self._get_feature_cols(df)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        X = pipeline.fit_transform(df[self._feature_cols])
        self._pipeline = pipeline

        y = df[self.TARGET_COL].values if self.TARGET_COL in df.columns else None
        logger.info(f"BETHPreprocessor fit: {X.shape[1]} features, {len(X)} samples")
        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self._pipeline is None:
            raise RuntimeError("Call fit_transform first")
        X = self._pipeline.transform(df[self._feature_cols])
        y = df[self.TARGET_COL].values if self.TARGET_COL in df.columns else None
        return X, y

    @property
    def feature_names(self) -> List[str]:
        return self._feature_cols or []

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "BETHPreprocessor":
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# Cyber Attacks Preprocessor
# ─────────────────────────────────────────────────────────────────────────────
class CyberAttacksPreprocessor:
    """
    Preprocessing for the synthetic Cybersecurity Attacks dataset.

    Supports two experiment modes:
    - 'metadata_only'  — realistic features available at packet-capture time
    - 'with_leakage'   — includes higher-level features (severity, indicators)
    """

    TARGET_COL = "Attack Type"
    FEATURE_SETS = {
        "metadata_only": [
            "Destination Port", "Protocol", "Packet Length",
            "Packet Type", "Traffic Type", "Anomaly Scores", "Action Taken"
        ],
        "with_leakage": [
            "Destination Port", "Protocol", "Packet Length",
            "Packet Type", "Traffic Type", "Anomaly Scores", "Action Taken",
            "Severity Level", "Malware Indicators"
        ],
    }

    def __init__(self, feature_set: str = "metadata_only"):
        if feature_set not in self.FEATURE_SETS:
            raise ValueError(f"feature_set must be one of {list(self.FEATURE_SETS)}")
        self.feature_set = feature_set
        self._pipeline: Optional[Pipeline] = None
        self._label_encoder = LabelEncoder()
        self._feature_cols: Optional[List[str]] = None

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        desired = self.FEATURE_SETS[self.feature_set]
        available = [c for c in desired if c in df.columns]
        if not available:
            raise ValueError("None of the expected feature columns found in DataFrame")
        self._feature_cols = available

        num_cols = [c for c in available if df[c].dtype != object]
        cat_cols = [c for c in available if df[c].dtype == object]

        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols))

        ct = ColumnTransformer(transformers, remainder="drop")
        self._pipeline = Pipeline([("ct", ct)])
        X = self._pipeline.fit_transform(df)

        y = self._label_encoder.fit_transform(df[self.TARGET_COL])
        logger.info(
            f"CyberAttacksPreprocessor ({self.feature_set}): "
            f"{X.shape[1]} features, {len(y)} samples"
        )
        return X, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = self._pipeline.transform(df)
        y = self._label_encoder.transform(df[self.TARGET_COL])
        return X, y

    @property
    def classes(self) -> List[str]:
        return list(self._label_encoder.classes_)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "CyberAttacksPreprocessor":
        return joblib.load(path)
