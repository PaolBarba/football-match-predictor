"""Pytest configuration file."""

import pytest
from pgptsql.oracle_client import OracleSettings


@pytest.fixture
def _oracle_env(monkeypatch):
    """MQTT Environment Variables for tests."""
    monkeypatch.setenv("ORACLE_USER", "env_user")
    monkeypatch.setenv("ORACLE_PASSWORD", "env_password")
    monkeypatch.setenv("ORACLE_DSN", "env_dsn")
    monkeypatch.setenv("ORACLE_TNS_ADMIN", "env_tns_admin")


@pytest.fixture
def oracle_settings():
    """Oracle settings fixture."""
    return OracleSettings(user="test_user", password="test_password", dsn="test_dsn", tns_admin="test_tns_admin")  # noqa: S106
