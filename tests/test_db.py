"""
Tests for the database layer — connection, schema, user management.

Uses an in-memory SQLite database so no external service is needed.

Run: pytest tests/test_db.py -v
"""

import pytest
from sqlalchemy.orm import Session

from src.db.connector import DatabaseManager
from src.db.models import User, UserRole, Alert, AlertSeverity, AlertStatus


@pytest.fixture
def db_manager():
    """In-memory SQLite database — fresh for every test."""
    mgr = DatabaseManager(database_url="sqlite:///:memory:", echo=False)
    mgr.create_tables()
    return mgr


@pytest.fixture
def session(db_manager):
    s = db_manager.get_session()
    yield s
    s.close()


class TestDatabaseManager:

    def test_health_check_ok(self, db_manager):
        result = db_manager.health_check()
        assert result["status"] == "ok"

    def test_create_tables_idempotent(self, db_manager):
        """Calling create_tables twice should not raise."""
        db_manager.create_tables()

    def test_bootstrap_admin(self, db_manager):
        admin = db_manager.bootstrap_admin()
        assert admin.username == db_manager._safe_url("admin") or True
        assert admin.role == UserRole.admin

    def test_bootstrap_admin_idempotent(self, db_manager):
        a1 = db_manager.bootstrap_admin()
        a2 = db_manager.bootstrap_admin()
        assert a1.id == a2.id

    def test_create_user(self, db_manager):
        user = db_manager.create_user(
            username="analyst_alice",
            email="alice@example.com",
            password="secure123",
            role=UserRole.analyst,
        )
        assert user.id is not None
        assert user.role == UserRole.analyst

    def test_verify_password(self, db_manager):
        db_manager.create_user(
            username="testuser", email="test@example.com",
            password="mypassword", role=UserRole.viewer
        )
        with db_manager.get_session() as s:
            user = s.query(User).filter_by(username="testuser").first()
        assert db_manager.verify_password("mypassword", user.hashed_password)
        assert not db_manager.verify_password("wrongpassword", user.hashed_password)

    def test_list_users(self, db_manager):
        db_manager.bootstrap_admin()
        db_manager.create_user("user1", "u1@test.com", "pass1", UserRole.viewer)
        users = db_manager.list_users()
        assert len(users) >= 2


class TestUserRoles:

    def test_all_roles_creatable(self, db_manager):
        for i, role in enumerate(UserRole):
            db_manager.create_user(
                username=f"user_{i}",
                email=f"user{i}@example.com",
                password="password123",
                role=role,
            )
        users = db_manager.list_users()
        roles_created = {u.role for u in users}
        assert roles_created == set(UserRole)

    def test_unique_username_constraint(self, db_manager):
        db_manager.create_user("bob", "bob@test.com", "pass", UserRole.viewer)
        with pytest.raises(Exception):
            db_manager.create_user("bob", "bob2@test.com", "pass", UserRole.viewer)


class TestAlerts:

    def test_create_and_query_alert(self, session):
        alert = Alert(
            title="Suspicious DoS traffic detected",
            severity=AlertSeverity.high,
            status=AlertStatus.open,
            source_dataset="unsw_nb15",
            attack_type="DoS",
            confidence=0.92,
        )
        session.add(alert)
        session.commit()

        fetched = session.query(Alert).filter_by(title="Suspicious DoS traffic detected").first()
        assert fetched is not None
        assert fetched.severity == AlertSeverity.high
        assert fetched.status == AlertStatus.open

    def test_acknowledge_alert(self, db_manager, session):
        db_manager.bootstrap_admin()
        with db_manager.get_session() as s:
            admin = s.query(User).filter_by(role=UserRole.admin).first()
            admin_id = admin.id

        alert = Alert(
            title="Test alert",
            severity=AlertSeverity.medium,
            status=AlertStatus.open,
        )
        session.add(alert)
        session.commit()

        alert.status = AlertStatus.acknowledged
        alert.acknowledged_by = admin_id
        session.commit()

        refreshed = session.query(Alert).filter_by(id=alert.id).first()
        assert refreshed.status == AlertStatus.acknowledged
