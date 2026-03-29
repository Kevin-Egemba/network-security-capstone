"""
Network Security Analytics Platform
====================================
A production-grade ML platform for network intrusion detection,
behavioral anomaly analysis, and threat classification.

Modules
-------
src.config          : Centralized configuration & path management
src.data.loader     : Dataset loading with validation
src.data.validator  : Schema and data quality checks
src.data.preprocessor : Feature engineering pipelines
src.models.supervised   : Supervised classification models
src.models.unsupervised : Anomaly detection & clustering
src.db.connector    : Database connection management (SQLite / PostgreSQL)
src.db.ingest       : ETL ingestion scripts
src.api.app         : FastAPI prediction service
"""

__version__ = "1.0.0"
__author__ = "Network Security Capstone"
