"""Tests for the A/B testing MCP tools (session scaffold / edit / run).

These exercise the filesystem + validation paths only — no model calls or network. The live
run paths (run_ab_session / run_model_ab against real providers) are credential-gated and return
an {"error": ...} dict, asserted here without hitting the network.
"""

import json

import pytest

mcp_server = pytest.importorskip("nodewatch.mcp_server")  # requires the optional `mcp` package


@pytest.fixture
def sessions_base(tmp_path, monkeypatch):
    monkeypatch.setenv("NODEWATCH_SESSIONS_DIR", str(tmp_path))
    return tmp_path


def test_init_ab_session_creates_config(sessions_base):
    r = mcp_server.init_ab_session("t1", transport="model")
    assert "error" not in r
    assert r["session_dir"] == str(sessions_base / "testing_sessions" / "t1")
    assert r["config"]["transport"] == "model"
    assert (sessions_base / "testing_sessions" / "t1" / "config.json").exists()


def test_init_ab_session_refuses_existing(sessions_base):
    mcp_server.init_ab_session("t2", transport="model")
    r = mcp_server.init_ab_session("t2", transport="model")
    assert "error" in r
    # force overwrites with a different transport
    r2 = mcp_server.init_ab_session("t2", transport="http", force=True)
    assert r2["config"]["transport"] == "http"


def test_read_then_write_edits_config(sessions_base):
    mcp_server.init_ab_session("t3", transport="model")
    cfg = mcp_server.read_ab_config("t3")["config"]

    cfg["questions"] = [{"id": "q1", "text": "Explain entropy."}]
    cfg["models"] = [
        {"id": "opus-4-8", "request_model": "claude-opus-4-8", "expect": "opus-4-8"},
        {"id": "sonnet-4-6", "request_model": "claude-sonnet-4-6", "expect": "sonnet-4-6"},
    ]
    w = mcp_server.write_ab_config("t3", cfg)
    assert "error" not in w

    persisted = json.loads((sessions_base / "testing_sessions" / "t3" / "config.json").read_text())
    assert [q["text"] for q in persisted["questions"]] == ["Explain entropy."]
    assert [m["id"] for m in persisted["models"]] == ["opus-4-8", "sonnet-4-6"]


def test_write_ab_config_rejects_invalid(sessions_base):
    bad = {"transport": "model", "models": [], "questions": []}
    r = mcp_server.write_ab_config("t4", bad)
    assert "error" in r
    # nothing written
    assert not (sessions_base / "testing_sessions" / "t4" / "config.json").exists()


def test_read_missing_session_errors(sessions_base):
    assert "error" in mcp_server.read_ab_config("does-not-exist")


def test_run_ab_session_missing_config_errors(sessions_base):
    assert "error" in mcp_server.run_ab_session("never-created")


# ── pause_check / two-step confirmation ───────────────────────────────────────


def test_run_ab_session_pause_check_requires_confirm(sessions_base):
    mcp_server.init_ab_session("gated", transport="model")
    cfg = mcp_server.read_ab_config("gated")["config"]
    cfg["experiment"]["pause_check"] = "Spend real money?"
    mcp_server.write_ab_config("gated", cfg)

    # First call: no confirm → confirmation_required, nothing run (no DB written).
    r = mcp_server.run_ab_session("gated")
    assert r["status"] == "confirmation_required"
    assert r["message"] == "Spend real money?"
    assert r["preview"]["n_models"] == 2
    assert not (sessions_base / "testing_sessions" / "gated" / "runs.db").exists()


def test_run_model_ab_pause_check_writes_config_but_not_run(sessions_base, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = mcp_server.run_model_ab(
        prompts=["a", "b"], models=["claude-opus-4-8", "claude-opus-4-7"],
        session="gm", pause_check=True,
    )
    assert r["status"] == "confirmation_required"
    assert r["preview"]["total_runs"] == 4            # 2 models × 2 prompts × 1 rep
    sess = sessions_base / "testing_sessions" / "gm"
    assert (sess / "config.json").exists()            # config written for inspection
    assert not (sess / "runs.db").exists()            # but nothing run yet
