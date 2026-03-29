"""
FastAPI prediction service for the Network Security Analytics Platform.

Endpoints
---------
POST /auth/token          — obtain JWT access token
GET  /health              — service health check
GET  /users/me            — current user info
POST /users               — create new user (admin only)
GET  /datasets            — list registered datasets
GET  /model-runs          — list training experiments
POST /predict/network     — predict attack from network flow features
POST /predict/bulk        — batch prediction (up to 1000 records)
GET  /alerts              — list open threat alerts
POST /alerts/{id}/ack     — acknowledge an alert
GET  /analytics/summary   — dataset + model performance summary

Run
---
    uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session

from src.config import db_settings, Paths
from src.db.connector import DatabaseManager, get_db
from src.db.models import Alert, AlertStatus, ModelRun, User, UserRole


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Network Security Analytics API",
    description=(
        "ML-powered network intrusion detection and threat classification. "
        "Trained on UNSW-NB15, BETH, and synthetic cybersecurity datasets."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# ── Model cache (loaded on demand) ────────────────────────────────────────────
_model_cache: Dict[str, Any] = {}


def _load_model(name: str) -> Any:
    if name not in _model_cache:
        path = Paths.MODELS / f"{name}.pkl"
        if not path.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model '{name}' not found. Train it first via the notebooks."
            )
        _model_cache[name] = joblib.load(path)
    return _model_cache[name]


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class Token(BaseModel):
    access_token: str
    token_type: str


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: UserRole = UserRole.viewer


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: UserRole
    is_active: bool
    created_at: Optional[datetime]

    class Config:
        from_attributes = True


class NetworkFlowInput(BaseModel):
    """Feature vector matching UNSW-NB15 schema (key features only)."""
    proto: str = Field("tcp", description="Protocol (tcp/udp/icmp)")
    service: str = Field("-", description="Service type")
    state: str = Field("FIN", description="Connection state")
    dur: float = Field(0.0, ge=0.0, description="Duration (seconds)")
    sbytes: int = Field(0, ge=0, description="Source-to-dest bytes")
    dbytes: int = Field(0, ge=0, description="Dest-to-source bytes")
    sttl: int = Field(64, ge=0, le=255, description="Source TTL")
    dttl: int = Field(64, ge=0, le=255, description="Dest TTL")
    spkts: int = Field(1, ge=0, description="Source packet count")
    dpkts: int = Field(1, ge=0, description="Dest packet count")


class PredictionOut(BaseModel):
    is_attack: bool
    attack_probability: float
    attack_type: Optional[str]
    confidence: float
    model_version: str = "1.0.0"


class AlertOut(BaseModel):
    id: int
    title: str
    severity: str
    status: str
    attack_type: Optional[str]
    confidence: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


# ── Auth helpers ──────────────────────────────────────────────────────────────
def _create_token(data: dict, expires_delta: timedelta | None = None) -> str:
    payload = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    payload.update({"exp": expire})
    return jwt.encode(payload, db_settings.secret_key, algorithm=db_settings.algorithm)


def _get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, db_settings.secret_key,
                             algorithms=[db_settings.algorithm])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = db.query(User).filter_by(username=username, is_active=True).first()
    if not user:
        raise credentials_exc
    return user


def _require_role(*roles: UserRole):
    def checker(current_user: User = Depends(_get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of: {[r.value for r in roles]}"
            )
        return current_user
    return checker


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    mgr = DatabaseManager()
    return {"status": "ok", "db": mgr.health_check(), "version": "1.0.0"}


@app.post("/auth/token", response_model=Token, tags=["Auth"])
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Obtain a JWT access token."""
    user = db.query(User).filter_by(username=form_data.username).first()
    if not user or not _pwd_context.verify(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = _create_token(
        {"sub": user.username},
        expires_delta=timedelta(minutes=db_settings.access_token_expire_minutes),
    )
    # Update last_login
    user.last_login = datetime.utcnow()
    db.commit()
    return {"access_token": token, "token_type": "bearer"}


@app.get("/users/me", response_model=UserOut, tags=["Users"])
def get_me(current_user: User = Depends(_get_current_user)):
    return current_user


@app.post("/users", response_model=UserOut, tags=["Users"],
          dependencies=[Depends(_require_role(UserRole.admin))])
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    """Create a new user account (admin only)."""
    if db.query(User).filter_by(username=payload.username).first():
        raise HTTPException(400, "Username already exists")
    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=_pwd_context.hash(payload.password),
        role=payload.role,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"New user created: {user.username} ({user.role})")
    return user


@app.get("/users", response_model=List[UserOut], tags=["Users"],
         dependencies=[Depends(_require_role(UserRole.admin))])
def list_users(db: Session = Depends(get_db)):
    return db.query(User).all()


@app.get("/datasets", tags=["Data"])
def list_datasets(db: Session = Depends(get_db),
                  _: User = Depends(_get_current_user)):
    from src.db.models import DatasetRegistry
    datasets = db.query(DatasetRegistry).all()
    return [
        {"id": d.id, "name": d.name, "rows": d.row_count,
         "ingested_at": d.ingested_at, "description": d.description}
        for d in datasets
    ]


@app.get("/model-runs", tags=["Models"])
def list_model_runs(
    limit: int = 20,
    db: Session = Depends(get_db),
    _: User = Depends(_get_current_user),
):
    runs = db.query(ModelRun).order_by(ModelRun.created_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id, "run_name": r.run_name, "model_type": r.model_type,
            "dataset": r.dataset_name, "algorithm": r.algorithm,
            "roc_auc": r.roc_auc, "f1_weighted": r.f1_weighted,
            "created_at": r.created_at,
        }
        for r in runs
    ]


@app.post("/predict/network", response_model=PredictionOut, tags=["Predict"])
def predict_network(
    payload: NetworkFlowInput,
    _: User = Depends(_get_current_user),
):
    """
    Predict whether a network flow is an attack.
    Returns binary prediction + attack type if flagged.
    """
    # Build feature array matching training schema
    feature_row = np.array([[
        payload.dur, payload.sbytes, payload.dbytes,
        payload.sttl, payload.dttl, payload.spkts, payload.dpkts,
    ]])

    try:
        model = _load_model("two_stage_detector")
        preprocessor = _load_model("unsw_preprocessor")
        # If preprocessor available, transform
        # (simplified path — full pipeline in notebooks)
        result = model.stage1_model.predict_proba(feature_row)[0]
        attack_proba = float(result[1])
        is_attack = attack_proba >= 0.5
        return PredictionOut(
            is_attack=is_attack,
            attack_probability=round(attack_proba, 4),
            attack_type="Unknown" if is_attack else None,
            confidence=round(max(result), 4),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")


@app.get("/alerts", response_model=List[AlertOut], tags=["Alerts"])
def get_alerts(
    status_filter: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    _: User = Depends(_get_current_user),
):
    q = db.query(Alert)
    if status_filter:
        q = q.filter(Alert.status == status_filter)
    return q.order_by(Alert.created_at.desc()).limit(limit).all()


@app.post("/alerts/{alert_id}/ack", tags=["Alerts"])
def acknowledge_alert(
    alert_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(_require_role(UserRole.admin, UserRole.analyst)),
):
    alert = db.query(Alert).filter_by(id=alert_id).first()
    if not alert:
        raise HTTPException(404, "Alert not found")
    alert.status = AlertStatus.acknowledged
    alert.acknowledged_at = datetime.utcnow()
    alert.acknowledged_by = current_user.id
    db.commit()
    return {"message": f"Alert {alert_id} acknowledged by {current_user.username}"}


@app.get("/analytics/summary", tags=["Analytics"])
def analytics_summary(
    db: Session = Depends(get_db),
    _: User = Depends(_get_current_user),
):
    """High-level platform summary for the dashboard."""
    from src.db.models import DatasetRegistry, NetworkEvent, SystemCallEvent
    from sqlalchemy import func as sqlfunc

    n_datasets = db.query(DatasetRegistry).count()
    n_network = db.query(NetworkEvent).count()
    n_syscall = db.query(SystemCallEvent).count()
    n_alerts_open = db.query(Alert).filter_by(status=AlertStatus.open).count()
    n_model_runs = db.query(ModelRun).count()
    best_auc = db.query(sqlfunc.max(ModelRun.roc_auc)).scalar()

    return {
        "datasets_registered": n_datasets,
        "network_events_stored": n_network,
        "syscall_events_stored": n_syscall,
        "open_alerts": n_alerts_open,
        "model_runs": n_model_runs,
        "best_roc_auc": best_auc,
    }
