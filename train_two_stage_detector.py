"""
Train and save the UNSW-NB15 TwoStageDetector — the actual reproducible path.

notebooks/03_unsw_supervised.ipynb explores individual algorithms inline but
never calls TwoStageDetector.fit() or saves models/two_stage_detector.pkl —
that artifact was previously produced by hand, outside any committed code
path, and Stage 2 (attack-type classification) was never actually fitted as
a result. This script is the source of truth for that artifact.

Stage 2 needs real attack_cat labels, not the binary label — TwoStageDetector
.fit()'s default path (no X_train_multi/y_multi passed) trains Stage 2 on
y_binary[attack_mask], which is all 1s after masking to attack-only rows.
That's a degenerate single-class fit. This script passes the real attack
categories explicitly so Stage 2 does what its name says.

Feature scope: trained on exactly the 10 fields NetworkFlowInput (the API)
and the "Live Detection" form actually collect — proto/service/state +
dur/sbytes/dbytes/sttl/dttl/spkts/dpkts — not the full 45-column research
schema. That's a deliberate choice, not a shortcut: it's what makes the
live-serving model actually consistent with what the live form can send it.
A full-feature research model would score higher and is a legitimate next
step, but it isn't what /predict/network or the dashboard call today.

Usage
-----
    python train_two_stage_detector.py
"""

from __future__ import annotations

import time

from loguru import logger

from src.config import Paths
from src.data.loader import DataLoader
from src.data.preprocessor import UNSWPreprocessor
from src.data.validator import DataValidator
from src.db.connector import DatabaseManager
from src.db.models import ModelRun
from src.models.supervised import TwoStageDetector

# Exactly what NetworkFlowInput (API) and the dashboard's Live Detection
# form collect — see module docstring for why training is scoped to this.
LIVE_FEATURE_COLS = [
    "proto", "service", "state",
    "dur", "sbytes", "dbytes", "sttl", "dttl", "spkts", "dpkts",
]


def main() -> None:
    loader = DataLoader()
    df_train, df_test = loader.load_unsw_nb15()

    validator = DataValidator()
    report = validator.validate_unsw(df_train, split="train")
    if not report.passed:
        raise RuntimeError(f"UNSW-NB15 training data failed validation: {report.errors}")

    # attack_cat is dropped by the preprocessor — grab it now, aligned to
    # df_train's row order (fit_transform doesn't reorder rows).
    attack_cat_train = df_train["attack_cat"].values

    live_cols = LIVE_FEATURE_COLS + ["label"]
    df_train_live = df_train[live_cols]
    df_test_live = df_test[live_cols]

    prep = UNSWPreprocessor()
    X_train, y_train = prep.fit_transform(df_train_live, target="label")
    X_test, y_test = prep.transform(df_test_live, target="label")

    attack_mask = y_train == 1
    X_train_multi = X_train[attack_mask]
    y_multi = attack_cat_train[attack_mask]
    logger.info(f"Stage 2 training set: {len(y_multi):,} attack rows, "
                f"{len(set(y_multi))} categories")

    detector = TwoStageDetector(stage1_model="xgb", stage2_model="rf")

    t0 = time.time()
    fit_results = detector.fit(
        X_train, y_train, X_train_multi=X_train_multi, y_multi=y_multi, cv=True
    )
    duration = time.time() - t0

    metrics = detector.evaluate(X_test, y_test, bootstrap=True)
    logger.info(
        f"Stage 1 — accuracy={metrics['accuracy']}  roc_auc={metrics['roc_auc']}  "
        f"95% CI=[{metrics['roc_auc_ci']['lower']}, {metrics['roc_auc_ci']['upper']}]"
    )

    detector.save(Paths.MODELS / "two_stage_detector.pkl")
    prep.save(Paths.MODELS / "unsw_preprocessor.pkl")

    mgr = DatabaseManager()
    mgr.create_tables()
    with mgr.get_session() as session:
        session.add(ModelRun(
            run_name=f"two_stage_detector_{int(time.time())}",
            model_type="supervised",
            dataset_name="UNSW-NB15",
            algorithm=f"{detector.stage1_name}+{detector.stage2_name} (two-stage)",
            hyperparameters={"stage1": detector.stage1_name, "stage2": detector.stage2_name,
                              "feature_cols": LIVE_FEATURE_COLS},
            accuracy=metrics["accuracy"],
            roc_auc=metrics["roc_auc"],
            f1_weighted=metrics["f1_weighted"],
            extra_metrics={
                "roc_auc_ci": metrics["roc_auc_ci"],
                "f1_ci": metrics["f1_ci"],
                "stage1_cv_auc": fit_results.get("stage1_cv_auc"),
                "stage2_categories": sorted(set(y_multi.tolist())),
            },
            model_artifact_path=str(Paths.MODELS / "two_stage_detector.pkl"),
            preprocessor_path=str(Paths.MODELS / "unsw_preprocessor.pkl"),
            duration_seconds=round(duration, 1),
            notes="Trained via train_two_stage_detector.py with real attack_cat labels "
                  "for Stage 2, scoped to the 10 fields the live API/dashboard form "
                  "actually collect (see module docstring).",
        ))
        session.commit()

    logger.info("Model run recorded in database. Training complete.")


if __name__ == "__main__":
    main()
