# Network Security Analytics Platform

A production-grade ML platform for **network intrusion detection**, **behavioral anomaly analysis**, and **threat classification** — demonstrating the complete data lifecycle from raw CSV ingestion through database storage, model training, REST API serving, and interactive dashboard.

Built as a graduate capstone project, this platform intentionally spans the full stack of skills expected from a **Data Analyst**, **Data Scientist**, and **Data Engineer**.

---

## Architecture

```
Raw CSVs  →  DataLoader  →  DataValidator  →  Preprocessor
                                                    ↓
                               Supervised Models (UNSW-NB15)
                               Unsupervised Models (BETH)
                                                    ↓
                              SQLite / PostgreSQL Database
                              ┌───────────────┐
                              │  ORM Models   │  users, network_events,
                              │  (SQLAlchemy) │  system_call_events,
                              │               │  model_runs, alerts
                              └───────────────┘
                               ↙                ↘
                     FastAPI REST API      Streamlit Dashboard
                     (JWT auth,            (6 pages: overview,
                      predictions,          threats, models,
                      alert mgmt)           live detection, SQL,
                                            user management)
```

---

## Datasets

| Dataset | Type | Rows | Task | Key Finding |
|---------|------|------|------|-------------|
| **UNSW-NB15** | Network flow telemetry | 257,673 | Binary detection + 9-class attack classification | ~94% AUC with two-stage RF/XGBoost pipeline |
| **BETH** | Honeypot system-call telemetry | ~100K | Unsupervised anomaly detection | Sparse labels require unsupervised approach; Isolation Forest best recall |
| **Cyber Attacks (Synthetic)** | Synthetic attack records | 40,000 | 3-class classification | Network metadata alone yields ~33% accuracy (near random) — demonstrates honest data assessment |

---

## Techniques Demonstrated

### Data Engineering
- Chunked ETL ingestion (`src/db/ingest.py`) with provenance tracking
- SQLAlchemy ORM with full schema (`src/db/models.py`)
- Role-based user management with bcrypt-hashed passwords
- Database migrations support via Alembic
- Docker Compose deployment (PostgreSQL + API + Dashboard)

### Data Analysis
- Exploratory data analysis with leakage detection (`src/data/validator.py`)
- SQL analytics views (`data/schemas/schema.sql`)
- Interactive SQL query interface in dashboard

### Data Science
- Modular preprocessing pipelines (`src/data/preprocessor.py`) — fit on train, applied to test, serializable
- **Supervised:** Two-stage IDS pipeline (detect → classify) with SMOTE, cross-validation, feature importance
- **Unsupervised:** 5-algorithm ensemble (K-Means, DBSCAN, Isolation Forest, GMM, PCA reconstruction error)
- MLflow experiment tracking integration
- Honest evaluation of data sufficiency (synthetic dataset reveals metadata limitations)

### ML Engineering
- FastAPI REST service with JWT auth (`src/api/app.py`)
- Model serialization with joblib
- Pytest test suite with 30+ tests across pipeline, DB, and integration scenarios

---

## Project Structure

```
Network Security Capstone/
│
├── notebooks/                     # Analysis notebooks (numbered workflow)
│   ├── 01_data_overview.ipynb
│   ├── 02_beth_unsupervised.ipynb
│   ├── 03_unsw_supervised.ipynb
│   ├── 04_cyber_attacks_analysis.ipynb
│   └── 05_results_comparison.ipynb
│
├── src/                           # Production Python library
│   ├── config.py                  # Centralized paths + settings
│   ├── data/
│   │   ├── loader.py              # Dataset loading
│   │   ├── validator.py           # Schema + leakage checks
│   │   └── preprocessor.py        # Feature engineering pipelines
│   ├── models/
│   │   ├── supervised.py          # TwoStageDetector, AttackClassifier
│   │   └── unsupervised.py        # AnomalyDetector (5 algorithms)
│   ├── db/
│   │   ├── models.py              # SQLAlchemy ORM tables
│   │   ├── connector.py           # DB connection + user management
│   │   └── ingest.py              # ETL ingestion scripts
│   └── api/
│       └── app.py                 # FastAPI prediction service
│
├── dashboard/
│   └── app.py                     # Streamlit multi-page analytics dashboard
│
├── data/
│   ├── raw/                       # Original, untouched data
│   ├── processed/                 # Engineered features
│   ├── schemas/schema.sql         # Database DDL
│   ├── Beth/                      # BETH honeypot dataset
│   ├── unsw_nb15/                 # UNSW-NB15 network telemetry
│   └── Cyber_Attacks/             # Synthetic attack dataset
│
├── configs/
│   ├── model_config.yaml          # Model hyperparameters
│   └── db_config.yaml             # Database + role configuration
│
├── tests/
│   ├── test_data_pipeline.py      # Validator + preprocessor tests
│   └── test_db.py                 # Database + user management tests
│
├── models/                        # Serialized model artifacts (git-ignored)
├── reports/figures/               # Saved visualizations
│
├── requirements.txt
├── docker-compose.yml
├── Dockerfile
└── .env.example
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — at minimum set SECRET_KEY
```

### 3. Initialize database
```bash
python -m src.db.connector --create
```

### 4. Ingest datasets
```bash
python -m src.db.ingest --all
# Or selectively:
python -m src.db.ingest --dataset unsw
python -m src.db.ingest --dataset beth
```

### 5. Run notebooks (in order)
```bash
jupyter lab
# Open notebooks/01_data_overview.ipynb and run through 05
```

### 6. Launch dashboard
```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

### 7. Start the API
```bash
uvicorn src.api.app:app --reload
# API docs at http://localhost:8000/docs
```

### 8. Run tests
```bash
pytest tests/ -v
```

---

## Docker Deployment

```bash
# Start all services (API + Dashboard + PostgreSQL)
docker compose up -d

# With pgAdmin for database inspection
docker compose --profile dev up -d

# Services:
#   http://localhost:8501  — Streamlit dashboard
#   http://localhost:8000  — FastAPI (docs at /docs)
#   http://localhost:5432  — PostgreSQL
#   http://localhost:5050  — pgAdmin (dev profile only)
```

---

## API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/token` | None | Get JWT token |
| `GET` | `/health` | None | Service health |
| `GET` | `/users/me` | JWT | Current user |
| `POST` | `/users` | Admin | Create user |
| `GET` | `/datasets` | JWT | List ingested datasets |
| `GET` | `/model-runs` | JWT | Experiment history |
| `POST` | `/predict/network` | JWT | Predict attack from flow features |
| `GET` | `/alerts` | JWT | List threat alerts |
| `POST` | `/alerts/{id}/ack` | Analyst/Admin | Acknowledge alert |
| `GET` | `/analytics/summary` | JWT | Platform KPIs |

Full interactive docs: `http://localhost:8000/docs`

---

## Key Results

### UNSW-NB15 (Two-Stage Pipeline)
| Stage | Model | AUC | F1 |
|-------|-------|-----|-----|
| Binary detection | Random Forest | 0.94 | 0.93 |
| Binary detection | XGBoost | 0.94 | 0.93 |
| Attack classification (9 classes) | RF + SMOTE | — | 0.81 |

### BETH Unsupervised Anomaly Detection
| Algorithm | Silhouette | Recall (evil) |
|-----------|-----------|---------------|
| K-Means (k=5) | 0.38 | 0.45 |
| Isolation Forest | — | 0.52 |
| DBSCAN | 0.31 | 0.48 |
| GMM | — | 0.41 |
| PCA Recon. Error | — | 0.39 |

### Cybersecurity Attacks (Critical Finding)
| Feature Set | Accuracy | Interpretation |
|-------------|----------|---------------|
| Metadata only | 33-34% | Random baseline! |
| With leakage | 33-34% | Still insufficient |

> **This is a feature, not a bug.** A model that honestly reports data insufficiency is more valuable than one that overfits noise. Real SOC systems need multiple data sources for fine-grained classification.

---

## User Roles

| Role | Permissions |
|------|------------|
| `admin` | Full access — manage users, models, data |
| `analyst` | Read, query, acknowledge alerts, view dashboard |
| `data_scientist` | Train models, run experiments, view results |
| `viewer` | Read-only dashboard access |

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Data processing | pandas, numpy, scikit-learn |
| ML models | scikit-learn, XGBoost, imbalanced-learn |
| Database | SQLAlchemy ORM, SQLite (dev) / PostgreSQL (prod) |
| API | FastAPI, Pydantic, JWT (python-jose) |
| Dashboard | Streamlit, Plotly |
| Experiment tracking | MLflow |
| Testing | pytest |
| Containerization | Docker, Docker Compose |
| Config | PyYAML, python-dotenv |
