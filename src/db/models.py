"""
SQLAlchemy ORM models — the full database schema for the platform.

Tables
------
users               : User accounts with role-based access
network_events      : UNSW-NB15 network flow records
system_call_events  : BETH system call telemetry
model_runs          : Experiment tracking (hyperparams, metrics)
predictions         : Per-record model predictions
alerts              : Generated threat alerts
dataset_registry    : Metadata about ingested datasets

Run `python -m src.db.connector --create` to create all tables.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


# ── Enums ─────────────────────────────────────────────────────────────────────
class UserRole(str, enum.Enum):
    admin = "admin"
    analyst = "analyst"
    data_scientist = "data_scientist"
    viewer = "viewer"


class AlertSeverity(str, enum.Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"


class AlertStatus(str, enum.Enum):
    open = "open"
    acknowledged = "acknowledged"
    resolved = "resolved"
    false_positive = "false_positive"


# ─────────────────────────────────────────────────────────────────────────────
# Users
# ─────────────────────────────────────────────────────────────────────────────
class User(Base):
    """
    User accounts with role-based access control.
    Roles: admin, analyst, data_scientist, viewer
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.viewer, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    last_login = Column(DateTime, nullable=True)

    # Relationships
    alerts_acknowledged = relationship("Alert", back_populates="acknowledged_by_user",
                                       foreign_keys="Alert.acknowledged_by")
    model_runs = relationship("ModelRun", back_populates="created_by_user")

    def __repr__(self):
        return f"<User(username={self.username!r}, role={self.role})>"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Registry
# ─────────────────────────────────────────────────────────────────────────────
class DatasetRegistry(Base):
    """Metadata about each ingested dataset — tracks provenance."""
    __tablename__ = "dataset_registry"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    source_file = Column(String(500))
    row_count = Column(Integer)
    column_count = Column(Integer)
    ingested_at = Column(DateTime, default=func.now())
    schema_version = Column(String(20), default="1.0.0")
    description = Column(Text)
    extra_metadata = Column(JSON)

    def __repr__(self):
        return f"<Dataset(name={self.name!r}, rows={self.row_count})>"


# ─────────────────────────────────────────────────────────────────────────────
# Network Events (UNSW-NB15)
# ─────────────────────────────────────────────────────────────────────────────
class NetworkEvent(Base):
    """
    Ingested UNSW-NB15 network flow records.
    Indexed on label and attack_cat for fast analytics queries.
    """
    __tablename__ = "network_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("dataset_registry.id"))
    split = Column(String(10))                 # train | test

    # Key features (subset of 45)
    proto = Column(String(20))
    service = Column(String(20))
    state = Column(String(20))
    dur = Column(Float)
    sbytes = Column(Integer)
    dbytes = Column(Integer)
    sttl = Column(Integer)
    dttl = Column(Integer)
    sloss = Column(Integer)
    dloss = Column(Integer)
    spkts = Column(Integer)
    dpkts = Column(Integer)
    ct_srv_src = Column(Integer)
    ct_dst_ltm = Column(Integer)
    ct_src_dport_ltm = Column(Integer)

    # Targets
    label = Column(Integer, index=True)        # 0=normal, 1=attack
    attack_cat = Column(String(30), index=True)

    ingested_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_network_label_cat", "label", "attack_cat"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# System Call Events (BETH)
# ─────────────────────────────────────────────────────────────────────────────
class SystemCallEvent(Base):
    """BETH honeypot system-call telemetry with sparse evil labels."""
    __tablename__ = "system_call_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(Integer, ForeignKey("dataset_registry.id"))
    split = Column(String(20))

    # Features
    process_id = Column(Integer)
    thread_id = Column(Integer)
    parent_process_id = Column(Integer)
    user_id = Column(Integer)
    mount_namespace = Column(Integer)
    event_id = Column(Integer, index=True)
    args_num = Column(Integer)
    return_value = Column(Integer)

    # Target
    evil = Column(Integer, index=True)         # 0=normal, 1=evil (sparse)

    ingested_at = Column(DateTime, default=func.now())

    __table_args__ = (
        Index("ix_syscall_evil_event", "evil", "event_id"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model Runs (Experiment Tracking)
# ─────────────────────────────────────────────────────────────────────────────
class ModelRun(Base):
    """
    Tracks every training experiment — hyperparameters, metrics, artifacts.
    Mirrors MLflow but embedded in our own DB for dashboard integration.
    """
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True)
    run_name = Column(String(100), nullable=False)
    model_type = Column(String(50))            # TwoStageDetector | AnomalyDetector | etc.
    dataset_name = Column(String(100))
    algorithm = Column(String(50))

    # Hyperparameters stored as JSON
    hyperparameters = Column(JSON)

    # Metrics
    accuracy = Column(Float)
    roc_auc = Column(Float)
    f1_weighted = Column(Float)
    f1_macro = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    extra_metrics = Column(JSON)

    # Artifact paths
    model_artifact_path = Column(String(500))
    preprocessor_path = Column(String(500))

    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=func.now())
    duration_seconds = Column(Float)
    notes = Column(Text)

    created_by_user = relationship("User", back_populates="model_runs")
    predictions = relationship("Prediction", back_populates="model_run")

    def __repr__(self):
        return f"<ModelRun(name={self.run_name!r}, auc={self.roc_auc})>"


# ─────────────────────────────────────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────────────────────────────────────
class Prediction(Base):
    """Stores individual model predictions for audit and drift tracking."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"))
    event_source = Column(String(20))          # network | syscall | synthetic
    source_record_id = Column(Integer)         # FK to original event (loose)

    predicted_label = Column(Integer)          # 0=normal, 1=attack
    predicted_class = Column(String(50))       # attack type if available
    confidence = Column(Float)                 # probability of predicted class
    true_label = Column(Integer, nullable=True)

    predicted_at = Column(DateTime, default=func.now())

    model_run = relationship("ModelRun", back_populates="predictions")


# ─────────────────────────────────────────────────────────────────────────────
# Alerts
# ─────────────────────────────────────────────────────────────────────────────
class Alert(Base):
    """
    Threat alerts generated from model predictions.
    Designed to mirror a real SOC ticketing workflow.
    """
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    description = Column(Text)
    severity = Column(Enum(AlertSeverity), default=AlertSeverity.medium)
    status = Column(Enum(AlertStatus), default=AlertStatus.open, index=True)

    # What triggered this alert
    source_dataset = Column(String(50))
    attack_type = Column(String(50))
    confidence = Column(Float)
    model_run_id = Column(Integer, ForeignKey("model_runs.id"), nullable=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)

    # Lifecycle
    created_at = Column(DateTime, default=func.now(), index=True)
    acknowledged_at = Column(DateTime, nullable=True)
    resolved_at = Column(DateTime, nullable=True)
    acknowledged_by = Column(Integer, ForeignKey("users.id"), nullable=True)

    acknowledged_by_user = relationship("User", back_populates="alerts_acknowledged",
                                         foreign_keys=[acknowledged_by])

    def __repr__(self):
        return f"<Alert(title={self.title!r}, severity={self.severity}, status={self.status})>"
