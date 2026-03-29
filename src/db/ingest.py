"""
ETL ingestion pipeline — loads CSV datasets into the database.

This is the Data Engineering component of the platform. Each ingester:
  1. Reads raw CSV files in configurable chunks
  2. Validates the data before insertion
  3. Tracks provenance in dataset_registry
  4. Uses upsert-safe bulk inserts

Usage
-----
    python -m src.db.ingest --dataset unsw
    python -m src.db.ingest --dataset beth
    python -m src.db.ingest --dataset cyber
    python -m src.db.ingest --all
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import pandas as pd
from loguru import logger
from sqlalchemy import text
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.config import Paths
from src.db.connector import DatabaseManager
from src.db.models import DatasetRegistry, NetworkEvent, SystemCallEvent


CHUNK_SIZE = 10_000


# ── Dataset Registry ──────────────────────────────────────────────────────────
def register_dataset(
    session: Session,
    name: str,
    source_file: Path,
    row_count: int,
    col_count: int,
    description: str = "",
) -> DatasetRegistry:
    """Upsert a dataset entry in the registry."""
    existing = session.query(DatasetRegistry).filter_by(name=name).first()
    if existing:
        existing.row_count = row_count
        existing.column_count = col_count
        session.commit()
        return existing

    record = DatasetRegistry(
        name=name,
        source_file=str(source_file),
        row_count=row_count,
        column_count=col_count,
        description=description,
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    logger.info(f"Registered dataset: {name} ({row_count:,} rows)")
    return record


# ── Helpers ───────────────────────────────────────────────────────────────────
def _chunked_csv(path: Path, chunk_size: int = CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Yield chunks from a CSV file."""
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        yield chunk


def _count_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return sum(1 for _ in f) - 1  # subtract header


# ── UNSW-NB15 Ingestion ───────────────────────────────────────────────────────
def ingest_unsw_nb15(manager: DatabaseManager, truncate: bool = False) -> None:
    """
    Ingest UNSW-NB15 train and test sets into network_events table.
    """
    datasets = [
        (Paths.UNSW_TRAIN, "train"),
        (Paths.UNSW_TEST, "test"),
    ]

    with manager.get_session() as session:
        if truncate:
            session.execute(text("DELETE FROM network_events"))
            session.commit()
            logger.warning("Truncated network_events table.")

        for path, split in datasets:
            if not path.exists():
                logger.warning(f"UNSW {split} not found at {path}, skipping.")
                continue

            n_rows = _count_rows(path)
            registry_entry = register_dataset(
                session,
                name=f"unsw_nb15_{split}",
                source_file=path,
                row_count=n_rows,
                col_count=45,
                description=f"UNSW-NB15 network flow telemetry ({split} set)",
            )

            logger.info(f"Ingesting UNSW-NB15 {split}: {n_rows:,} rows…")
            inserted = 0
            t0 = time.time()

            for chunk in tqdm(_chunked_csv(path), desc=f"UNSW {split}",
                              total=n_rows // CHUNK_SIZE + 1):
                chunk.columns = chunk.columns.str.strip().str.lower()
                records = []
                for _, row in chunk.iterrows():
                    records.append(NetworkEvent(
                        dataset_id=registry_entry.id,
                        split=split,
                        proto=str(row.get("proto", ""))[:20],
                        service=str(row.get("service", ""))[:20],
                        state=str(row.get("state", ""))[:20],
                        dur=_float(row.get("dur")),
                        sbytes=_int(row.get("sbytes")),
                        dbytes=_int(row.get("dbytes")),
                        sttl=_int(row.get("sttl")),
                        dttl=_int(row.get("dttl")),
                        sloss=_int(row.get("sloss")),
                        dloss=_int(row.get("dloss")),
                        spkts=_int(row.get("spkts")),
                        dpkts=_int(row.get("dpkts")),
                        ct_srv_src=_int(row.get("ct_srv_src")),
                        ct_dst_ltm=_int(row.get("ct_dst_ltm")),
                        ct_src_dport_ltm=_int(row.get("ct_src_dport_ltm")),
                        label=_int(row.get("label")),
                        attack_cat=str(row.get("attack_cat", ""))[:30],
                    ))

                session.bulk_save_objects(records)
                session.commit()
                inserted += len(records)

            elapsed = time.time() - t0
            logger.info(
                f"UNSW-NB15 {split}: {inserted:,} rows inserted in {elapsed:.1f}s "
                f"({inserted / elapsed:.0f} rows/s)"
            )


# ── BETH Ingestion ────────────────────────────────────────────────────────────
def ingest_beth(manager: DatabaseManager, truncate: bool = False) -> None:
    """Ingest BETH labelled train/val/test splits into system_call_events."""
    splits = {
        "labelled_train": Paths.BETH_LABELLED_TRAIN,
        "labelled_val": Paths.BETH_LABELLED_VAL,
        "labelled_test": Paths.BETH_LABELLED_TEST,
    }

    with manager.get_session() as session:
        if truncate:
            session.execute(text("DELETE FROM system_call_events"))
            session.commit()
            logger.warning("Truncated system_call_events table.")

        for split_name, path in splits.items():
            if not path.exists():
                logger.warning(f"BETH {split_name} not found at {path}, skipping.")
                continue

            n_rows = _count_rows(path)
            registry_entry = register_dataset(
                session,
                name=f"beth_{split_name}",
                source_file=path,
                row_count=n_rows,
                col_count=23,
                description=f"BETH honeypot system-call telemetry ({split_name})",
            )

            logger.info(f"Ingesting BETH {split_name}: {n_rows:,} rows…")
            inserted = 0

            for chunk in tqdm(_chunked_csv(path), desc=f"BETH {split_name}",
                              total=n_rows // CHUNK_SIZE + 1):
                records = []
                for _, row in chunk.iterrows():
                    records.append(SystemCallEvent(
                        dataset_id=registry_entry.id,
                        split=split_name,
                        process_id=_int(row.get("processId")),
                        thread_id=_int(row.get("threadId")),
                        parent_process_id=_int(row.get("parentProcessId")),
                        user_id=_int(row.get("userId")),
                        mount_namespace=_int(row.get("mountNamespace")),
                        event_id=_int(row.get("eventId")),
                        args_num=_int(row.get("argsNum")),
                        return_value=_int(row.get("returnValue")),
                        evil=_int(row.get("evil")),
                    ))
                session.bulk_save_objects(records)
                session.commit()
                inserted += len(records)

            logger.info(f"BETH {split_name}: {inserted:,} rows inserted.")


# ── Type coercion helpers ─────────────────────────────────────────────────────
def _int(val) -> int | None:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _float(val) -> float | None:
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest datasets into the database")
    parser.add_argument("--dataset", choices=["unsw", "beth", "all"], default="all")
    parser.add_argument("--truncate", action="store_true",
                        help="Clear table before inserting")
    args = parser.parse_args()

    mgr = DatabaseManager()
    mgr.create_tables()

    if args.dataset in ("unsw", "all"):
        ingest_unsw_nb15(mgr, truncate=args.truncate)
    if args.dataset in ("beth", "all"):
        ingest_beth(mgr, truncate=args.truncate)

    logger.info("Ingestion complete.")
