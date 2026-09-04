"""Tests for the JSON-config A/B runner (abrun.load_ab_config / run_ab_config)."""

import json
from pathlib import Path

import pytest

from nodewatch import abrun
from nodewatch.models import LLMCall, NodeSpan, RunTrace
from nodewatch.storage.sqlite import SQLiteStorage

_HTTP_CONFIG = {
    "transport": "http",
    "api": {
        "url": "http://localhost:8000/v1/query",
        "headers": {"Content-Type": "application/json", "Authorization": "Bearer ${AB_TEST_TOKEN}"},
        "model_field": "model",
        "prompt_field": "user_prompt",
        "conversation_id_field": "conversation_id",
        "body": {"stream": False},
    },
    "experiment": {"reps": 1, "settle_seconds": 0, "switch_mode": "per_request"},
    "models": [
        {"id": "opus-4-8", "request_model": "claude-opus-4-8", "expect": "opus-4-8"},
        {"id": "opus-4-7", "request_model": "claude-opus-4-7", "expect": "opus-4-7"},
    ],
    "questions": [{"id": "q1", "text": "What is the capital of France?"}],
}


# ── config parsing ───────────────────────────────────────────────────────────


def test_load_ab_config_parses_http(tmp_path):
    path = tmp_path / "ab.json"
    path.write_text(json.dumps(_HTTP_CONFIG))
    cfg = abrun.load_ab_config(path)

    assert cfg.transport == "http"
    assert [m.id for m in cfg.models] == ["opus-4-8", "opus-4-7"]
    assert [m.request_model for m in cfg.models] == ["claude-opus-4-8", "claude-opus-4-7"]
    assert cfg.api.model_field == "model"
    assert cfg.reps == 1
    assert len(cfg.questions) == 1 and cfg.questions[0].id == "q1"


def test_per_request_requires_model_field():
    bad = {**_HTTP_CONFIG, "api": {"url": "http://x"}}  # no model_field
    with pytest.raises(ValueError, match="model_field"):
        abrun.parse_ab_config(bad)


def test_model_transport_rejects_manual_switch():
    bad = {
        "transport": "model",
        "model": {"provider": "anthropic"},
        "experiment": {"switch_mode": "manual"},
        "models": [{"id": "a", "expect": "opus-4-8"}],
        "questions": [{"id": "q1", "text": "hi"}],
    }
    with pytest.raises(ValueError, match="manual"):
        abrun.parse_ab_config(bad)


def test_model_transport_parses_into_model_config():
    cfg = abrun.parse_ab_config({
        "transport": "model",
        "model": {"provider": "anthropic", "system": "be brief", "max_tokens": 256},
        "models": [{"id": "a", "request_model": "claude-opus-4-8", "expect": "opus-4-8"}],
        "questions": [{"id": "q1", "text": "hi"}],
    })
    assert cfg.transport == "model"
    assert cfg.model.provider == "anthropic"
    assert cfg.model.system == "be brief"
    assert cfg.model.max_tokens == 256


# ── HTTP query_fn shape + env expansion ────────────────────────────────────────


def test_build_http_query_fn_posts_expected_body(monkeypatch, tmp_path):
    monkeypatch.setenv("AB_TEST_TOKEN", "s3cret")
    path = tmp_path / "ab.json"
    path.write_text(json.dumps(_HTTP_CONFIG))
    cfg = abrun.load_ab_config(path)

    captured = {}

    class _Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b""

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        return _Resp()

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    query_fn = abrun.build_http_query_fn(cfg.api, "claude-opus-4-8")
    query_fn("What is the capital of France?", "ab_opus-4-8_q1_r1", extra="x")

    body = captured["body"]
    assert body["user_prompt"] == "What is the capital of France?"
    assert body["conversation_id"] == "ab_opus-4-8_q1_r1"
    assert body["model"] == "claude-opus-4-8"      # model_field injection
    assert body["stream"] is False                  # static body field
    assert body["extra"] == "x"                     # kwargs pass-through
    # ${AB_TEST_TOKEN} expanded at load time, before the header reached urllib
    auth = {k.lower(): v for k, v in captured["headers"].items()}["authorization"]
    assert auth == "Bearer s3cret"


# ── end-to-end run_ab_config with a fake transport ─────────────────────────────


_SERVED = {"claude-opus-4-8": "us.anthropic.claude-opus-4-8",
           "claude-opus-4-7": "us.anthropic.claude-opus-4-7"}


def _fake_build_query_fn(storage):
    """Return a build_query_fn replacement that writes a synthetic tagged run per call."""
    def build(config, model_id, store):
        def qf(text, conversation_id, **kw):
            served = _SERVED.get(model_id, "us.anthropic.claude-opus-4-8")
            span = NodeSpan(
                node_name="coordinator", start_time=0.0, end_time=1.0,
                llm_calls=[LLMCall(node_name="coordinator", model=served, provider="bedrock",
                                   input_tokens=1000, output_tokens=200, stop_reason="end_turn")],
            )
            store.save(RunTrace(
                graph_name="v2", query=text, total_duration_ms=1000.0, node_spans=[span],
                metadata={"conversation_id": conversation_id, "ab_question_id": kw.get("ab_question_id", "")},
            ))
        return qf
    return build


def test_run_ab_config_end_to_end(tmp_db, monkeypatch):
    storage = SQLiteStorage(tmp_db)
    monkeypatch.setattr(abrun, "build_query_fn", _fake_build_query_fn(storage))
    cfg = abrun.parse_ab_config(_HTTP_CONFIG)

    result = abrun.run_ab_config(cfg, storage)

    assert result["comparison"] is not None
    assert result["comparison"]["verified_ok"] is True
    assert len(result["comparison"]["per_question"]) == 1
    assert "opus-4-8 vs opus-4-7" in result["summary"]
    storage.close()


def test_run_ab_config_manual_mode_calls_pause_hook(tmp_db, monkeypatch):
    storage = SQLiteStorage(tmp_db)
    monkeypatch.setattr(abrun, "build_query_fn", _fake_build_query_fn(storage))
    manual_cfg = {
        "transport": "http",
        "api": {"url": "http://x", "prompt_field": "user_prompt"},
        "experiment": {"reps": 1, "settle_seconds": 0, "switch_mode": "manual"},
        "models": [{"id": "m48", "expect": "opus-4-8"}, {"id": "m47", "expect": "opus-4-7"}],
        "questions": [{"id": "q1", "text": "What is the capital of France?"}],
    }
    cfg = abrun.parse_ab_config(manual_cfg)

    paused = []
    abrun.run_ab_config(cfg, storage, pause_hook=lambda phase: paused.append(phase.name))

    assert paused == ["m48", "m47"]    # one pause per phase, in order
    storage.close()


def test_run_ab_config_writes_one_report_per_agent(tmp_db, tmp_path, monkeypatch):
    storage = SQLiteStorage(tmp_db)
    monkeypatch.setattr(abrun, "build_query_fn", _fake_build_query_fn(storage))
    cfg = abrun.parse_ab_config(_HTTP_CONFIG)
    out_dir = tmp_path / "reports"

    result = abrun.run_ab_config(cfg, storage, out_dir=out_dir)

    paths = result["report_paths"]
    assert sorted(p.split("/")[-1] for p in paths) == ["ab_opus-4-7.json", "ab_opus-4-8.json"]

    rep = json.loads((out_dir / "ab_opus-4-8.json").read_text())
    assert rep["model"] == "opus-4-8"
    assert rep["verified"] is True
    assert rep["n_ok"] == 1
    q = rep["questions"][0]
    # every field the user asked for: question, time, tokens, nodes, final answer
    assert q["question"] == "What is the capital of France?"
    assert q["duration_ms"] > 0
    assert q["total_tokens"] == 1200
    assert q["nodes_called"] == ["coordinator"]
    assert "final_answer" in q
    storage.close()


def test_run_ab_config_three_models_no_comparison(tmp_db, monkeypatch):
    storage = SQLiteStorage(tmp_db)
    monkeypatch.setattr(abrun, "build_query_fn", _fake_build_query_fn(storage))
    cfg = abrun.parse_ab_config({
        "transport": "http",
        "api": {"url": "http://x", "model_field": "model"},
        "experiment": {"reps": 1, "settle_seconds": 0, "switch_mode": "per_request"},
        "models": [
            {"id": "a", "request_model": "claude-opus-4-8", "expect": "opus-4-8"},
            {"id": "b", "request_model": "claude-opus-4-7", "expect": "opus-4-7"},
            {"id": "c", "request_model": "claude-opus-4-8", "expect": "opus-4-8"},
        ],
        "questions": [{"id": "q1", "text": "hi"}],
    })

    result = abrun.run_ab_config(cfg, storage)
    assert result["comparison"] is None             # ≠2 models → no pairwise comparison
    assert len(result["model_summaries"]) == 3
    storage.close()


# ── model-transport credential gate (no network) ───────────────────────────────


def test_model_transport_credential_gate(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = abrun.parse_ab_config({
        "transport": "model",
        "model": {"provider": "anthropic"},
        "models": [{"id": "a", "request_model": "claude-opus-4-8", "expect": "opus-4-8"}],
        "questions": [{"id": "q1", "text": "hi"}],
    })
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        abrun.build_model_query_fn(cfg.model, "claude-opus-4-8", None)


# ── testing sessions ─────────────────────────────────────────────────────────


def test_resolve_session_dir_bare_name_vs_path(monkeypatch, tmp_path):
    monkeypatch.setenv("NODEWATCH_SESSIONS_DIR", str(tmp_path))
    # bare name → <base>/testing_sessions/<name>
    assert abrun.resolve_session_dir("opus48-vs-47") == tmp_path / "testing_sessions" / "opus48-vs-47"
    # explicit absolute path → as-is
    assert abrun.resolve_session_dir("/tmp/my-test") == Path("/tmp/my-test")


def test_init_session_template_parses(tmp_path):
    config_path = abrun.init_session(tmp_path / "s1", transport="model")
    assert config_path == tmp_path / "s1" / "config.json"
    cfg = abrun.load_ab_config(config_path)           # template must be a valid config
    assert cfg.transport == "model"
    assert len(cfg.models) == 2 and len(cfg.questions) == 2


def test_init_session_http_template_parses(tmp_path):
    cfg = abrun.load_ab_config(abrun.init_session(tmp_path / "s2", transport="http"))
    assert cfg.transport == "http"
    assert cfg.api.model_field == "model"             # per_request-ready


def test_init_session_from_file_copies(tmp_path):
    src = tmp_path / "seed.json"
    src.write_text(json.dumps(_HTTP_CONFIG))
    cfg = abrun.load_ab_config(abrun.init_session(tmp_path / "s3", from_file=src))
    assert [m.id for m in cfg.models] == ["opus-4-8", "opus-4-7"]


def test_init_session_refuses_overwrite(tmp_path):
    abrun.init_session(tmp_path / "s4", transport="model")
    with pytest.raises(FileExistsError):
        abrun.init_session(tmp_path / "s4", transport="model")
    # force overwrites
    abrun.init_session(tmp_path / "s4", transport="http", force=True)
    assert abrun.load_ab_config(tmp_path / "s4" / "config.json").transport == "http"


def test_config_db_field_parsed(monkeypatch):
    monkeypatch.setenv("ABTEST_DB", "/srv/runs.db")
    cfg = abrun.parse_ab_config({**_HTTP_CONFIG, "db": "${ABTEST_DB}"})
    assert cfg.db == "/srv/runs.db"                   # env-expanded


def test_pause_check_parsing():
    def parse(pc):
        raw = {**_HTTP_CONFIG, "experiment": {**_HTTP_CONFIG["experiment"], "pause_check": pc}}
        return abrun.parse_ab_config(raw).pause_check

    assert parse(False) is None                        # disabled
    assert parse(True) == abrun.DEFAULT_PAUSE_MESSAGE  # default message
    assert parse("Spend $5?") == "Spend $5?"           # custom message
    # absent → None
    assert abrun.parse_ab_config(_HTTP_CONFIG).pause_check is None


def test_preview_ab_config_shape():
    cfg = abrun.parse_ab_config({**_HTTP_CONFIG, "experiment": {**_HTTP_CONFIG["experiment"], "reps": 3}})
    pv = abrun.preview_ab_config(cfg)
    assert pv["n_models"] == 2 and pv["n_questions"] == 1 and pv["reps"] == 3
    assert pv["total_runs"] == 6                       # 2 models × 1 question × 3 reps
    assert pv["transport"] == "http"


def test_run_ab_config_writes_results_json(tmp_path, monkeypatch):
    session = tmp_path / "sess"
    session.mkdir()
    storage = SQLiteStorage(str(session / "runs.db"))
    monkeypatch.setattr(abrun, "build_query_fn", _fake_build_query_fn(storage))
    cfg = abrun.parse_ab_config(_HTTP_CONFIG)

    result = abrun.run_ab_config(cfg, storage, out_dir=str(session))

    results_path = session / "results.json"
    assert result["results_path"] == str(results_path)
    assert results_path.exists()
    loaded = json.loads(results_path.read_text())     # round-trips
    assert loaded["comparison"]["verified_ok"] is True
    assert "summary" in loaded
    # per-agent files alongside it
    assert (session / "ab_opus-4-8.json").exists()
    assert (session / "ab_opus-4-7.json").exists()
    storage.close()
