"""
Centralized configuration for the Network Security Analytics Platform.

Usage
-----
    from src.config import settings, Paths

    df = pd.read_csv(Paths.UNSW_TRAIN)
    seed = settings.random_seed
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml
from dotenv import load_dotenv

# Load .env if present (silently skips if missing)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

# ── Resolved project root ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]


# ── Static path registry ──────────────────────────────────────────────────────
class Paths:
    """All dataset and artifact paths relative to project root."""

    # Data directories
    DATA = ROOT / "data"
    RAW = DATA / "raw"
    PROCESSED = DATA / "processed"
    SCHEMAS = DATA / "schemas"

    # UNSW-NB15
    UNSW_TRAIN = DATA / "unsw_nb15" / "CSV Files" / "UNSW_NB15_training-set.csv"
    UNSW_TEST = DATA / "unsw_nb15" / "CSV Files" / "UNSW_NB15_testing-set.csv"

    # BETH
    BETH_MAIN = DATA / "Beth" / "BETH.csv"
    BETH_LABELLED_TRAIN = DATA / "Beth" / "labelled_training_data.csv"
    BETH_LABELLED_VAL = DATA / "Beth" / "labelled_validation_data.csv"
    BETH_LABELLED_TEST = DATA / "Beth" / "labelled_testing_data.csv"

    # Cybersecurity Attacks (synthetic)
    CYBER_ATTACKS = DATA / "Cyber_Attacks" / "cybersecurity_attacks.csv"

    # Artifacts
    MODELS = ROOT / "models"
    REPORTS = ROOT / "reports"
    FIGURES = ROOT / "reports" / "figures"
    LOGS = ROOT / "logs"

    # Configs
    MODEL_CONFIG = ROOT / "configs" / "model_config.yaml"
    DB_CONFIG = ROOT / "configs" / "db_config.yaml"

    @classmethod
    def ensure_dirs(cls) -> None:
        """Create artifact directories if they don't exist."""
        for attr in [cls.MODELS, cls.REPORTS, cls.FIGURES, cls.LOGS,
                     cls.PROCESSED]:
            attr.mkdir(parents=True, exist_ok=True)


# ── Settings loaded from YAML + environment ───────────────────────────────────
@dataclass
class ModelSettings:
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = -1
    primary_metric: str = "roc_auc"

    # Loaded from configs/model_config.yaml
    _raw: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if Paths.MODEL_CONFIG.exists():
            with open(Paths.MODEL_CONFIG) as f:
                self._raw = yaml.safe_load(f) or {}
            self.random_seed = self._raw.get("random_seed", self.random_seed)

    def get(self, key: str, default=None):
        return self._raw.get(key, default)


@dataclass
class DatabaseSettings:
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'network_security.db'}")
    )
    echo: bool = False
    pool_pre_ping: bool = True

    # JWT / Auth
    secret_key: str = field(
        default_factory=lambda: os.getenv("SECRET_KEY", "insecure-dev-key-change-in-production")
    )
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Admin bootstrap
    admin_username: str = field(default_factory=lambda: os.getenv("ADMIN_USERNAME", "admin"))
    admin_email: str = field(default_factory=lambda: os.getenv("ADMIN_EMAIL", "admin@example.com"))
    admin_password: str = field(default_factory=lambda: os.getenv("ADMIN_PASSWORD", "changeme"))


@dataclass
class MLflowSettings:
    tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns"))
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "network_security_capstone")
    )


# ── Singleton instances (import these) ────────────────────────────────────────
settings = ModelSettings()
db_settings = DatabaseSettings()
mlflow_settings = MLflowSettings()

# Ensure output directories exist on first import
Paths.ensure_dirs()
