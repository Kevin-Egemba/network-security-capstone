"""
DataLoader — Centralized dataset loading with validation and diagnostics.

All datasets are loaded through this module so path management, encoding
issues, and basic shape checks are handled in one place.

Usage
-----
    from src.data.loader import DataLoader

    loader = DataLoader()
    df_train, df_test = loader.load_unsw_nb15()
    df_beth = loader.load_beth(variant="labelled_train")
    df_attacks = loader.load_cyber_attacks()
    summary = loader.dataset_summary()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Optional, Tuple

import pandas as pd
from loguru import logger

from src.config import Paths


BETHVariant = Literal["main", "labelled_train", "labelled_val", "labelled_test"]

_BETH_PATH_MAP: dict[str, Path] = {
    "main": Paths.BETH_MAIN,
    "labelled_train": Paths.BETH_LABELLED_TRAIN,
    "labelled_val": Paths.BETH_LABELLED_VAL,
    "labelled_test": Paths.BETH_LABELLED_TEST,
}


class DataLoader:
    """
    Loads the three capstone datasets with consistent logging and basic
    integrity checks.

    Attributes
    ----------
    verbose : bool
        If True, log shape / dtype summaries after every load.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ── UNSW-NB15 ─────────────────────────────────────────────────────────────
    def load_unsw_nb15(
        self, *, drop_id: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load UNSW-NB15 training and test sets.

        Parameters
        ----------
        drop_id : bool
            Drop the `id` column (not a feature).

        Returns
        -------
        df_train, df_test : pd.DataFrame
        """
        self._check_path(Paths.UNSW_TRAIN, "UNSW-NB15 train")
        self._check_path(Paths.UNSW_TEST, "UNSW-NB15 test")

        df_train = pd.read_csv(Paths.UNSW_TRAIN, low_memory=False)
        df_test = pd.read_csv(Paths.UNSW_TEST, low_memory=False)

        for df, name in [(df_train, "train"), (df_test, "test")]:
            df.columns = df.columns.str.strip().str.lower()
            if drop_id and "id" in df.columns:
                df.drop(columns=["id"], inplace=True)

        if self.verbose:
            logger.info(f"UNSW-NB15 train: {df_train.shape}  test: {df_test.shape}")
            self._log_class_balance(df_train, "label", "UNSW train")

        return df_train, df_test

    # ── BETH ──────────────────────────────────────────────────────────────────
    def load_beth(self, variant: BETHVariant = "labelled_train") -> pd.DataFrame:
        """
        Load a BETH dataset variant.

        Parameters
        ----------
        variant : {"main", "labelled_train", "labelled_val", "labelled_test"}

        Returns
        -------
        pd.DataFrame
        """
        path = _BETH_PATH_MAP.get(variant)
        if path is None:
            raise ValueError(f"Unknown BETH variant '{variant}'. "
                             f"Choose from: {list(_BETH_PATH_MAP)}")

        self._check_path(path, f"BETH ({variant})")
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()

        if self.verbose:
            logger.info(f"BETH ({variant}): {df.shape}")
            if "evil" in df.columns:
                self._log_class_balance(df, "evil", f"BETH {variant}")

        return df

    # ── Cybersecurity Attacks (Synthetic) ─────────────────────────────────────
    def load_cyber_attacks(self) -> pd.DataFrame:
        """
        Load the synthetic cybersecurity attacks dataset.

        Returns
        -------
        pd.DataFrame  (40 000 rows × 25 columns)
        """
        self._check_path(Paths.CYBER_ATTACKS, "Cyber Attacks")
        df = pd.read_csv(Paths.CYBER_ATTACKS, low_memory=False)
        df.columns = df.columns.str.strip()

        if self.verbose:
            logger.info(f"Cyber Attacks: {df.shape}")
            self._log_class_balance(df, "Attack Type", "Cyber Attacks")

        return df

    # ── Dataset Summary ───────────────────────────────────────────────────────
    def dataset_summary(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame summarising all three datasets.

        Useful for the data overview notebook and the dashboard.
        """
        rows = []
        datasets = {
            "UNSW-NB15 (train)": (Paths.UNSW_TRAIN, "label"),
            "UNSW-NB15 (test)": (Paths.UNSW_TEST, "label"),
            "BETH (labelled_train)": (Paths.BETH_LABELLED_TRAIN, "evil"),
            "Cyber Attacks": (Paths.CYBER_ATTACKS, "Attack Type"),
        }

        for name, (path, target_col) in datasets.items():
            if not path.exists():
                rows.append({"dataset": name, "rows": None, "columns": None,
                             "target": target_col, "status": "file_missing"})
                continue
            df = pd.read_csv(path, low_memory=False, nrows=5)
            full = pd.read_csv(path, low_memory=False)
            rows.append({
                "dataset": name,
                "rows": len(full),
                "columns": len(full.columns),
                "target": target_col,
                "missing_pct": round(full.isnull().mean().mean() * 100, 2),
                "status": "ok",
            })

        return pd.DataFrame(rows)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _check_path(path: Path, label: str) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"[DataLoader] {label} not found at:\n  {path}\n"
                "Check that the data directory is intact."
            )

    @staticmethod
    def _log_class_balance(df: pd.DataFrame, col: str, label: str) -> None:
        if col not in df.columns:
            return
        counts = df[col].value_counts(normalize=True).mul(100).round(1)
        logger.info(f"  {label} class balance ({col}):\n{counts.to_string()}")
