"""Tests for optional API Bearer token authentication."""


import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_no_auth(tmp_path, monkeypatch):
    """Create app without NODEWATCH_API_TOKEN set."""
    monkeypatch.delenv("NODEWATCH_API_TOKEN", raising=False)
    monkeypatch.setenv("NODEWATCH_DB", str(tmp_path / "test.db"))

    from nodewatch.api import _create_standalone_app
    # Force fresh import to pick up env
    app = _create_standalone_app()
    return TestClient(app)


@pytest.fixture
def app_with_auth(tmp_path, monkeypatch):
    """Create app with NODEWATCH_API_TOKEN set."""
    monkeypatch.setenv("NODEWATCH_API_TOKEN", "secret-token-123")
    monkeypatch.setenv("NODEWATCH_DB", str(tmp_path / "test.db"))

    from nodewatch.api import _create_standalone_app
    app = _create_standalone_app()
    return TestClient(app)


def test_no_auth_allows_requests(app_no_auth):
    """Without NODEWATCH_API_TOKEN, all requests should succeed."""
    resp = app_no_auth.post("/", json={"method": "list_runs", "args": {}})
    assert resp.status_code == 200


def test_auth_rejects_missing_header(app_with_auth):
    """With NODEWATCH_API_TOKEN set, requests without Authorization should get 401."""
    resp = app_with_auth.post("/", json={"method": "list_runs", "args": {}})
    assert resp.status_code == 401


def test_auth_rejects_wrong_token(app_with_auth):
    """With NODEWATCH_API_TOKEN set, wrong token should get 401."""
    resp = app_with_auth.post(
        "/",
        json={"method": "list_runs", "args": {}},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_auth_accepts_correct_token(app_with_auth):
    """With NODEWATCH_API_TOKEN set, correct token should succeed."""
    resp = app_with_auth.post(
        "/",
        json={"method": "list_runs", "args": {}},
        headers={"Authorization": "Bearer secret-token-123"},
    )
    assert resp.status_code == 200
