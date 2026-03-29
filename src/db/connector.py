"""
Database connection manager.

Supports both SQLite (default, zero-config) and PostgreSQL (production).
Connection string comes from the DATABASE_URL environment variable.

Usage
-----
    # Context manager — auto-closes session
    from src.db.connector import DatabaseManager
    with DatabaseManager() as db:
        users = db.session.query(User).all()

    # FastAPI dependency injection
    from src.db.connector import get_db
    def my_endpoint(db: Session = Depends(get_db)):
        ...

    # Create all tables
    from src.db.connector import DatabaseManager
    DatabaseManager().create_tables()

    # Bootstrap admin user
    DatabaseManager().bootstrap_admin()
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from loguru import logger
from passlib.context import CryptContext
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.config import db_settings
from src.db.models import Base, User, UserRole

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class DatabaseManager:
    """
    Thin wrapper around SQLAlchemy engine + session factory.

    Parameters
    ----------
    database_url : str
        Override the URL from settings (useful for tests).
    echo : bool
        Log all SQL statements (debug mode).
    """

    def __init__(
        self,
        database_url: str | None = None,
        echo: bool = False,
    ):
        url = database_url or db_settings.database_url
        self.engine = create_engine(
            url,
            echo=echo,
            pool_pre_ping=db_settings.pool_pre_ping,
            connect_args={"check_same_thread": False} if "sqlite" in url else {},
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )
        logger.info(f"DatabaseManager connected → {self._safe_url(url)}")

    @staticmethod
    def _safe_url(url: str) -> str:
        """Mask password in logged URL."""
        if "@" in url:
            scheme, rest = url.split("://", 1)
            _, hostpart = rest.split("@", 1)
            return f"{scheme}://***@{hostpart}"
        return url

    # ── Table management ──────────────────────────────────────────────────────
    def create_tables(self) -> None:
        """Create all tables defined in models.py (idempotent)."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("All tables created (or already exist).")

    def drop_tables(self, confirm: bool = False) -> None:
        """Drop all tables. Requires explicit confirm=True."""
        if not confirm:
            raise RuntimeError("Pass confirm=True to drop all tables.")
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All tables dropped.")

    # ── Session management ────────────────────────────────────────────────────
    def get_session(self) -> Session:
        return self.SessionLocal()

    def __enter__(self) -> "DatabaseManager":
        self.session = self.SessionLocal()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type:
            self.session.rollback()
        else:
            self.session.commit()
        self.session.close()

    # ── Health check ──────────────────────────────────────────────────────────
    def health_check(self) -> dict:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {"status": "ok", "engine": str(self.engine.url)}
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    # ── User management ───────────────────────────────────────────────────────
    def bootstrap_admin(self) -> User:
        """
        Create the default admin user if it doesn't exist.
        Called on first startup from db_settings.
        """
        with self.get_session() as session:
            existing = session.query(User).filter_by(
                username=db_settings.admin_username
            ).first()
            if existing:
                logger.info(f"Admin user '{db_settings.admin_username}' already exists.")
                return existing

            admin = User(
                username=db_settings.admin_username,
                email=db_settings.admin_email,
                hashed_password=_pwd_context.hash(db_settings.admin_password),
                role=UserRole.admin,
                is_active=True,
            )
            session.add(admin)
            session.commit()
            session.refresh(admin)
            logger.info(f"Admin user '{admin.username}' created.")
            return admin

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.viewer,
    ) -> User:
        with self.get_session() as session:
            user = User(
                username=username,
                email=email,
                hashed_password=_pwd_context.hash(password),
                role=role,
                is_active=True,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            logger.info(f"User created: {username} ({role})")
            return user

    def verify_password(self, plain: str, hashed: str) -> bool:
        return _pwd_context.verify(plain, hashed)

    def list_users(self) -> list[User]:
        with self.get_session() as session:
            return session.query(User).all()


# ── FastAPI dependency ────────────────────────────────────────────────────────
_default_manager: DatabaseManager | None = None


def get_default_manager() -> DatabaseManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = DatabaseManager()
    return _default_manager


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a scoped DB session."""
    db = get_default_manager().SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    manager = DatabaseManager()
    if "--create" in sys.argv:
        manager.create_tables()
        manager.bootstrap_admin()
        print("Database initialised.")
    elif "--health" in sys.argv:
        print(manager.health_check())
    else:
        print("Usage: python -m src.db.connector --create | --health")
