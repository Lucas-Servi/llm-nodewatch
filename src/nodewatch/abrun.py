"""Run an A/B model benchmark from a JSON config file.

This is a thin, opt-in layer on top of the transport-agnostic :mod:`nodewatch.experiment`
runner. A JSON config describes WHERE to send each prompt (an HTTP endpoint, or a model
called directly) and WHICH models to compare; this module turns that into an
:class:`~nodewatch.experiment.ExperimentSpec`, builds the matching ``query_fn``, drives
the phases, and returns a render-ready comparison dict.

Two transports (JSON top-level ``"transport"``):

* ``"http"`` (default) — POST each prompt to an agent API via ``urllib`` (no ``requests``
  dependency). Field names (prompt / conversation_id / model) are config-driven so any
  request shape works. Use ``switch_mode: "per_request"`` when the API selects the model
  from the request body, or ``"manual"`` when the model is fixed at server startup (the
  runner pauses via ``pause_hook`` so an operator can reconfigure + restart between phases).
* ``"model"`` — call the model directly through its LangChain client (Anthropic / OpenAI /
  Bedrock), with a nodewatch tracker attached so the call is recorded to storage like any
  other run. The most self-contained option — just credentials, no external server — and it
  always selects the model per request. Provider packages and credentials are optional and
  lazily resolved; absence raises a clear error.

The config never contains secrets: ``${VAR}`` in the URL / headers / body is expanded from
the environment at load time, and model-transport credentials come from the standard env
(``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` / ``AWS_*``).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .experiment import ABExperiment, ExperimentQuestion, ExperimentResult, ExperimentSpec, Phase
from .stats import ABComparison, compute_ab_comparison, node_sig

if TYPE_CHECKING:
    from .models import RunTrace

logger = logging.getLogger(__name__)

QueryFn = Callable[..., Any]

_VALID_TRANSPORTS = ("http", "model")
_VALID_SWITCH_MODES = ("per_request", "manual")
_VALID_PROVIDERS = ("anthropic", "openai", "bedrock")

# Default confirmation message when ``pause_check`` is enabled without custom text.
DEFAULT_PAUSE_MESSAGE = "About to run the A/B test (this records runs; the 'model' transport calls the models and incurs API cost). Proceed?"


def _parse_pause_check(value: Any) -> str | None:
    """Normalize the ``pause_check`` config value to a message string or None.

    ``False``/absent → None (no gate); ``True`` → the default message; a non-empty string → that
    string (a custom confirmation prompt set when the test pipeline was created).
    """
    if value is None or value is False:
        return None
    if value is True:
        return DEFAULT_PAUSE_MESSAGE
    text = str(value).strip()
    return text or DEFAULT_PAUSE_MESSAGE


# ── config dataclasses ──────────────────────────────────────────────────────────


@dataclass
class ApiConfig:
    """HTTP transport settings (the ``"api"`` block)."""
    url: str
    method: str = "POST"
    headers: dict = field(default_factory=lambda: {"Content-Type": "application/json"})
    timeout_seconds: float = 600.0
    prompt_field: str = "user_prompt"
    conversation_id_field: str = "conversation_id"
    model_field: str | None = None          # when set → per-request model injection
    body: dict = field(default_factory=dict)  # static fields merged into every request


@dataclass
class ModelConfig:
    """Direct-model transport settings (the ``"model"`` block)."""
    provider: str = "anthropic"             # "anthropic" | "openai" | "bedrock"
    system: str | None = None
    max_tokens: int | None = None
    params: dict = field(default_factory=dict)  # extra kwargs for the client constructor


@dataclass
class ModelEntry:
    """One model under test — becomes one phase/cohort."""
    id: str                                 # short label / phase name, e.g. "opus-4-8"
    request_model: str                      # value sent to the API / passed to the client
    expect: str                             # served-model substring used to verify the cohort


@dataclass
class ABRunConfig:
    transport: str
    models: list[ModelEntry]
    questions: list[ExperimentQuestion]
    api: ApiConfig | None = None
    model: ModelConfig | None = None
    reps: int = 1
    settle_seconds: float = 3.0
    switch_mode: str = "per_request"
    conv_id_template: str = "ab_{phase}_{qid}_r{rep}"
    db: str | None = None               # optional DB path (e.g. the server's shared DB); --db overrides
    pause_check: str | None = None      # confirmation message to show before running (None = no gate)


# ── loading / validation ──────────────────────────────────────────────────────────


def _expand(value: Any) -> Any:
    """Expand ``${VAR}`` in strings (recursively for dict/list) from the environment."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"Invalid A/B config: {msg}")


def load_ab_config(path: str | Path) -> ABRunConfig:
    """Parse and validate a JSON A/B run config. Raises ``ValueError`` on a bad config."""
    raw = json.loads(Path(path).read_text())
    return parse_ab_config(raw)


def parse_ab_config(raw: dict) -> ABRunConfig:
    """Validate an already-parsed config dict into an :class:`ABRunConfig`."""
    transport = raw.get("transport", "http")
    _require(transport in _VALID_TRANSPORTS, f"transport must be one of {_VALID_TRANSPORTS}")

    exp = raw.get("experiment", {}) or {}
    switch_mode = exp.get("switch_mode", "per_request")
    _require(switch_mode in _VALID_SWITCH_MODES, f"switch_mode must be one of {_VALID_SWITCH_MODES}")

    # models → phases
    raw_models = raw.get("models") or []
    _require(len(raw_models) >= 1, "at least one entry in 'models' is required")
    models: list[ModelEntry] = []
    for m in raw_models:
        _require("id" in m, "each model needs an 'id'")
        _require("expect" in m, f"model '{m.get('id')}' needs an 'expect' (served-model substring)")
        models.append(ModelEntry(
            id=str(m["id"]),
            request_model=str(m.get("request_model", m["id"])),
            expect=str(m["expect"]),
        ))

    # questions
    raw_questions = raw.get("questions") or []
    _require(len(raw_questions) >= 1, "at least one entry in 'questions' is required")
    questions: list[ExperimentQuestion] = []
    for q in raw_questions:
        _require("id" in q and "text" in q, "each question needs an 'id' and 'text'")
        questions.append(ExperimentQuestion(id=str(q["id"]), text=str(q["text"]),
                                            kwargs=dict(q.get("kwargs") or {})))

    api = None
    model = None
    if transport == "http":
        raw_api = _expand(raw.get("api") or {})
        _require("url" in raw_api, "the 'api' block needs a 'url' for the http transport")
        headers = raw_api.get("headers") or {"Content-Type": "application/json"}
        api = ApiConfig(
            url=raw_api["url"],
            method=raw_api.get("method", "POST"),
            headers=headers,
            timeout_seconds=float(raw_api.get("timeout_seconds", 600.0)),
            prompt_field=raw_api.get("prompt_field", "user_prompt"),
            conversation_id_field=raw_api.get("conversation_id_field", "conversation_id"),
            model_field=raw_api.get("model_field"),
            body=dict(raw_api.get("body") or {}),
        )
        if switch_mode == "per_request":
            _require(bool(api.model_field),
                     "switch_mode 'per_request' needs api.model_field (the request key for the model); "
                     "use switch_mode 'manual' if the server's model is fixed at startup")
    else:  # model transport
        _require(switch_mode != "manual",
                 "the 'model' transport selects the model per request; switch_mode 'manual' "
                 "does not apply (omit it or set 'per_request')")
        raw_model = raw.get("model") or {}
        provider = raw_model.get("provider", "anthropic")
        _require(provider in _VALID_PROVIDERS, f"model.provider must be one of {_VALID_PROVIDERS}")
        model = ModelConfig(
            provider=provider,
            system=raw_model.get("system"),
            max_tokens=raw_model.get("max_tokens"),
            params=dict(raw_model.get("params") or {}),
        )

    db = raw.get("db")
    if db:
        db = os.path.expandvars(str(db))

    return ABRunConfig(
        transport=transport, models=models, questions=questions, api=api, model=model,
        reps=int(exp.get("reps", 1)),
        settle_seconds=float(exp.get("settle_seconds", 3.0)),
        switch_mode=switch_mode,
        conv_id_template=exp.get("conv_id_template", "ab_{phase}_{qid}_r{rep}"),
        db=db,
        pause_check=_parse_pause_check(exp.get("pause_check")),
    )


# ── testing sessions ────────────────────────────────────────────────────────────
# A "session" is a self-contained folder holding the config and every artifact of one
# A/B test: config.json (input), runs.db (recorded runs), ab_<model>.json (per-agent
# detail), and results.json (the full comparison). Point a command at the folder and
# everything loads from / dumps into that one place.

SESSIONS_DIRNAME = "testing_sessions"
CONFIG_FILENAME = "config.json"


def sessions_base_dir() -> Path:
    """Base dir under which bare session names live (``NODEWATCH_SESSIONS_DIR`` or ./)."""
    return Path(os.getenv("NODEWATCH_SESSIONS_DIR", ".")).expanduser()


def resolve_session_dir(name_or_path: str) -> Path:
    """Resolve a session reference to a folder path.

    A bare name (no path separator) lives under ``<base>/testing_sessions/<name>``; anything
    containing a separator — or an absolute path — is used as-is. So ``opus48-vs-47`` and
    ``/tmp/my-test`` (or ``./some/dir``) all work.
    """
    raw = str(name_or_path)
    p = Path(raw)
    if p.is_absolute() or (os.sep in raw) or (p.parts and len(p.parts) > 1) or raw.startswith("."):
        return p.expanduser()
    return sessions_base_dir() / SESSIONS_DIRNAME / raw


def default_config_template(transport: str = "model") -> dict:
    """A generic, ready-to-edit config for a new session (no environment-specific values)."""
    models = [
        {"id": "opus-4-8", "request_model": "claude-opus-4-8", "expect": "opus-4-8"},
        {"id": "opus-4-7", "request_model": "claude-opus-4-7", "expect": "opus-4-7"},
    ]
    questions = [
        {"id": "q1", "text": "What is the capital of France?"},
        {"id": "q2", "text": "Summarize the theory of relativity in two sentences."},
    ]
    if transport == "model":
        return {
            "transport": "model",
            "model": {"provider": "anthropic", "system": "You are a concise, helpful assistant.",
                      "max_tokens": 1024, "params": {"temperature": 0}},
            # pause_check: false | true | "your confirmation message" — gate the run on a user OK.
            "experiment": {"reps": 1, "settle_seconds": 0, "pause_check": False},
            "models": models,
            "questions": questions,
        }
    return {
        "transport": "http",
        "api": {
            "url": "http://localhost:8000/v1/query",
            "method": "POST",
            "headers": {"Content-Type": "application/json", "Authorization": "Bearer ${API_TOKEN}"},
            "timeout_seconds": 600,
            "model_field": "model",
            "prompt_field": "user_prompt",
            "conversation_id_field": "conversation_id",
            "body": {},
        },
        # pause_check: false | true | "your confirmation message" — gate the run on a user OK.
        "experiment": {"reps": 3, "settle_seconds": 3, "switch_mode": "per_request", "pause_check": False},
        "models": models,
        "questions": questions,
    }


def init_session(session_dir: str | Path, *, transport: str = "model",
                 from_file: str | Path | None = None, force: bool = False) -> Path:
    """Create a session folder with a ``config.json`` to edit. Returns the config path.

    ``from_file`` seeds the config by copying an existing config file; otherwise a generic
    template for ``transport`` is written. Refuses to overwrite an existing config unless ``force``.
    """
    out = Path(session_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / CONFIG_FILENAME
    if config_path.exists() and not force:
        raise FileExistsError(f"{config_path} already exists (use force=True / --force to overwrite)")
    if from_file is not None:
        raw = json.loads(Path(from_file).read_text())   # validate it parses as JSON
        config_path.write_text(json.dumps(raw, indent=2))
    else:
        config_path.write_text(json.dumps(default_config_template(transport), indent=2))
    return config_path


# ── transport builders ────────────────────────────────────────────────────────────


def build_http_query_fn(api: ApiConfig, model: str | None) -> QueryFn:
    """Build a ``query_fn`` that POSTs each prompt to ``api.url`` via ``urllib``.

    ``model`` is injected under ``api.model_field`` when both are set (per-request selection);
    pass ``None`` for the manual switch mode (model fixed server-side).
    """
    import urllib.request

    def query_fn(text: str, conversation_id: str, **kwargs: Any) -> None:
        body = dict(api.body)
        body[api.prompt_field] = text
        body[api.conversation_id_field] = conversation_id
        if api.model_field and model is not None:
            body[api.model_field] = model
        body.update(kwargs)
        data = json.dumps(body).encode()
        req = urllib.request.Request(api.url, data=data, method=api.method, headers=dict(api.headers))
        with urllib.request.urlopen(req, timeout=api.timeout_seconds) as resp:  # noqa: S310 (config-driven)
            resp.read()
        return None

    return query_fn


def _check_provider_credentials(provider: str) -> None:
    """Raise a friendly ``RuntimeError`` if credentials for ``provider`` are not available."""
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("set ANTHROPIC_API_KEY to use the 'anthropic' model transport")
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("set OPENAI_API_KEY to use the 'openai' model transport")
    if provider == "bedrock":
        try:
            import botocore.session
        except ImportError as e:
            raise RuntimeError("install boto3/botocore to use the 'bedrock' model transport") from e
        if botocore.session.Session().get_credentials() is None:
            raise RuntimeError(
                "no AWS credentials found (set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or a profile) "
                "to use the 'bedrock' model transport"
            )


def _build_chat_client(model_cfg: ModelConfig, model_id: str):
    """Construct the LangChain chat client for a provider (lazy import).

    Uses the provider classes directly (``ChatAnthropic`` / ``ChatOpenAI`` /
    ``ChatBedrockConverse``) rather than ``langchain.chat_models.init_chat_model`` — the
    latter lives in the ``langchain`` meta-package, which is not a nodewatch dependency.
    """
    params = dict(model_cfg.params)
    if model_cfg.max_tokens is not None:
        params.setdefault("max_tokens", model_cfg.max_tokens)
    try:
        if model_cfg.provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_id, **params)
        if model_cfg.provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_id, **params)
        if model_cfg.provider == "bedrock":
            from langchain_aws import ChatBedrockConverse
            return ChatBedrockConverse(model=model_id, **params)
    except ImportError as e:
        raise RuntimeError(
            f"install langchain-{ 'aws' if model_cfg.provider == 'bedrock' else model_cfg.provider } "
            f"to use the '{model_cfg.provider}' model transport"
        ) from e
    raise RuntimeError(f"unknown model provider {model_cfg.provider!r}; expected one of {_VALID_PROVIDERS}")


def build_model_query_fn(model_cfg: ModelConfig, model_id: str, storage) -> QueryFn:
    """Build a ``query_fn`` that calls ``model_id`` directly and records the run to storage.

    Gated on credentials/deps (raises a clear ``RuntimeError`` when missing). The single
    LLM call runs with a :class:`~nodewatch.tracker.GraphTracker` attached, so it is captured
    and persisted exactly like a graph run; the served model id lands in ``llm_calls.model``.
    """
    _check_provider_credentials(model_cfg.provider)
    client = _build_chat_client(model_cfg, model_id)

    def query_fn(text: str, conversation_id: str, **kwargs: Any) -> None:
        from .tracker import GraphTracker

        qid = kwargs.get("ab_question_id", "")
        tracker = GraphTracker(
            "model",
            metadata={"conversation_id": conversation_id, "ab_question_id": qid},
        )
        messages: list[tuple[str, str]] = []
        if model_cfg.system:
            messages.append(("system", model_cfg.system))
        messages.append(("human", text))
        # Pass langgraph_node via config metadata so the tracker assigns a real span
        # (otherwise the call lands in the filtered-out "__unknown__" bucket).
        cfg = {"callbacks": [tracker], "metadata": {"langgraph_node": "model"}}
        try:
            resp = client.invoke(messages, config=cfg)
        except Exception as e:
            trace = tracker.finalize(query=text, final_response="")
            if not trace.error:
                trace.error = repr(e)
            storage.save(trace)
            raise
        content = getattr(resp, "content", "")
        if isinstance(content, list):
            content = " ".join(str(p) for p in content)
        trace = tracker.finalize(query=text, final_response=str(content))
        storage.save(trace)
        return None

    return query_fn


def build_query_fn(config: ABRunConfig, model_id: str | None, storage) -> QueryFn:
    """Build the ``query_fn`` for one phase, dispatching on the configured transport."""
    if config.transport == "model":
        assert config.model is not None
        return build_model_query_fn(config.model, model_id or "", storage)
    assert config.api is not None
    return build_http_query_fn(config.api, model_id)


# ── driving + result shaping ────────────────────────────────────────────────────


def _build_spec(config: ABRunConfig) -> ExperimentSpec:
    questions: list[ExperimentQuestion] = []
    for q in config.questions:
        kw = dict(q.kwargs)
        if config.transport == "model":
            # Thread the question id into tracker metadata for deterministic pairing.
            kw.setdefault("ab_question_id", q.id)
        questions.append(ExperimentQuestion(id=q.id, text=q.text, kwargs=kw))
    phases = [Phase(name=m.id, expected_model=m.expect) for m in config.models]
    return ExperimentSpec(
        phases=phases, questions=questions, reps=config.reps,
        settle_seconds=config.settle_seconds, conv_id_template=config.conv_id_template,
    )


def comparison_to_dict(comp: ABComparison) -> dict:
    """Flatten an :class:`ABComparison` into the dict shape the CLI/API render."""
    return {
        "cohort_a": comp.cohort_a, "cohort_b": comp.cohort_b, "verified_ok": comp.verified_ok,
        "verification": [vars(v) for v in comp.verification],
        "per_question": [vars(q) for q in comp.per_question],
        "overall_duration_delta_pct": comp.overall_duration_delta_pct,
        "overall_tokens_delta_pct": comp.overall_tokens_delta_pct,
        "overall_filtered_per_call_a": comp.overall_filtered_per_call_a,
        "overall_filtered_per_call_b": comp.overall_filtered_per_call_b,
    }


def _model_summaries(result: ExperimentResult) -> list[dict]:
    """Per-model aggregates (used when there is no pairwise comparison)."""
    out = []
    for phase in result.spec.phases:
        recs = result.records_for_phase(phase.name)
        ok = [r for r in recs if r.ok]
        n_ok = len(ok)
        served = sorted({m for r in ok for m in r.served_models})
        out.append({
            "model": phase.name,
            "expected": phase.expected_model,
            "served_models": served,
            "n_runs": len(recs),
            "n_ok": n_ok,
            "mean_tokens": (sum(r.total_tokens for r in ok) / n_ok) if n_ok else 0.0,
            "mean_duration_ms": (sum(r.duration_ms for r in ok) / n_ok) if n_ok else 0.0,
        })
    return out


def _question_report(trace: RunTrace, record) -> dict:
    """Per-question detail for one run, pulled from the fully-hydrated trace."""
    nodes = []
    for s in trace.node_spans:
        nodes.append({
            "node": s.node_name,
            "type": s.node_type,
            "duration_ms": round(s.duration_ms, 1),
            "input_tokens": s.total_input_tokens,
            "output_tokens": s.total_output_tokens,
            "tool_calls": [tc.tool_name for tc in s.tool_calls],
        })
    return {
        "question_id": record.question_id,
        "rep": record.rep,
        "conversation_id": record.conversation_id,
        "run_id": trace.run_id,
        "question": trace.query,
        "ok": record.ok,
        "duration_ms": round(trace.total_duration_ms, 1),
        "total_tokens": trace.total_tokens,
        "input_tokens": trace.total_input_tokens,
        "output_tokens": trace.total_output_tokens,
        "cost_usd": round(trace.total_cost_usd, 6),
        "llm_calls": trace.total_llm_calls,
        "tool_calls": trace.total_tool_calls,
        "filtered": trace.total_filtered,
        "served_models": record.served_models,
        "node_path": node_sig(trace.nodes_visited),
        "nodes_called": trace.nodes_visited,
        "nodes": nodes,
        "final_answer": trace.final_response or "",
        "error": trace.error,
    }


def build_agent_reports(config: ABRunConfig, result: ExperimentResult) -> dict[str, dict]:
    """Build one self-contained report per model (agent) tested, keyed by model id.

    Each report carries every question's time / tokens / nodes / final answer plus per-agent
    aggregates — designed so two agents' files can be diffed directly.
    """
    reports: dict[str, dict] = {}
    for phase, entry in zip(result.spec.phases, config.models):
        recs = result.records_for_phase(phase.name)
        questions = []
        for r in recs:
            trace = result._traces_by_run.get(r.run_id) if r.run_id else None
            if trace is not None:
                questions.append(_question_report(trace, r))
            else:
                questions.append({
                    "question_id": r.question_id, "rep": r.rep,
                    "conversation_id": r.conversation_id, "run_id": r.run_id,
                    "ok": False, "error": r.api_error or "no run recorded",
                })
        ok = [q for q in questions if q.get("ok")]
        n_ok = len(ok)
        served = sorted({m for r in recs for m in r.served_models})
        verified = any(phase.expected_model in m for m in served)
        reports[phase.name] = {
            "model": phase.name,
            "request_model": entry.request_model,
            "expected_model": phase.expected_model,
            "served_models": served,
            "verified": verified,
            "transport": config.transport,
            "n_questions": len(questions),
            "n_ok": n_ok,
            "totals": {
                "total_tokens": sum(q.get("total_tokens", 0) for q in ok),
                "total_cost_usd": round(sum(q.get("cost_usd", 0.0) for q in ok), 6),
                "total_duration_ms": round(sum(q.get("duration_ms", 0.0) for q in ok), 1),
                "total_filtered": sum(q.get("filtered", 0) for q in ok),
                "mean_tokens": (sum(q.get("total_tokens", 0) for q in ok) / n_ok) if n_ok else 0.0,
                "mean_duration_ms": (sum(q.get("duration_ms", 0.0) for q in ok) / n_ok) if n_ok else 0.0,
            },
            "questions": questions,
        }
    return reports


def write_agent_reports(reports: dict[str, dict], out_dir: str | Path) -> list[str]:
    """Write one ``ab_<model>.json`` per agent into ``out_dir``. Returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for model, report in reports.items():
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in model)
        path = out / f"ab_{safe}.json"
        path.write_text(json.dumps(report, indent=2))
        paths.append(str(path))
    return paths


def preview_ab_config(config: ABRunConfig) -> dict:
    """Summarize what a run WILL do, for a pre-run confirmation gate (no side effects)."""
    total_runs = len(config.models) * len(config.questions) * max(1, config.reps)
    note = (
        "the 'model' transport calls the models directly and incurs real API cost"
        if config.transport == "model"
        else "posts each prompt to the configured HTTP API"
    )
    return {
        "transport": config.transport,
        "models": [m.id for m in config.models],
        "n_models": len(config.models),
        "n_questions": len(config.questions),
        "reps": config.reps,
        "total_runs": total_runs,
        "note": note,
    }


def _summary_line(result: ExperimentResult, comp: ABComparison | None) -> str:
    n = len(result.records)
    n_ok = sum(1 for r in result.records if r.ok)
    head = f"{len(result.spec.phases)} models, {n} runs ({n_ok} ok)"
    if comp is None:
        return head
    verdict = "verified" if comp.verified_ok else "NOT verified (served-model mismatch)"
    return (
        f"{head}; {comp.cohort_a} vs {comp.cohort_b} [{verdict}]: "
        f"duration {comp.overall_duration_delta_pct:+.0f}%, tokens {comp.overall_tokens_delta_pct:+.0f}%"
    )


def run_ab_config(config: ABRunConfig, storage, *, pause_hook: Callable[[Phase], None] | None = None,
                  out_dir: str | Path | None = None) -> dict:
    """Drive an A/B run from a parsed config and return a render-ready dict.

    For the ``http``/``manual`` switch mode, ``pause_hook`` is called before each phase so an
    operator can repoint the server at the phase's model + restart. When ``out_dir`` is given,
    one ``ab_<model>.json`` per agent (model) is written there — each holds every question's time,
    tokens, nodes, and final answer so the two files can be diffed. Returns a dict with
    ``comparison`` (the flat A/B dict, or ``None`` when not exactly 2 models), ``model_summaries``,
    ``records``, ``agent_reports``, ``report_paths``, and a one-line ``summary``.
    """
    spec = _build_spec(config)
    exp = ABExperiment(storage, spec)

    all_records = []
    all_traces: dict = {}
    for phase, entry in zip(spec.phases, config.models):
        if config.transport == "http" and config.switch_mode == "manual":
            if pause_hook is not None:
                pause_hook(phase)
            query_fn = build_query_fn(config, None, storage)
        else:
            query_fn = build_query_fn(config, entry.request_model, storage)
        recs, traces = exp.run_phase(phase, query_fn)
        all_records.extend(recs)
        all_traces.update(traces)

    result = ExperimentResult(spec=spec, records=all_records, _traces_by_run=all_traces)

    comp = None
    if len(spec.phases) == 2:
        pa, pb = spec.phases
        comp = compute_ab_comparison(
            result.traces_for_phase(pa.name), result.traces_for_phase(pb.name),
            expected_a=pa.expected_model, expected_b=pb.expected_model,
            cohort_a=pa.name, cohort_b=pb.name,
        )
        result.comparison = comp

    agent_reports = build_agent_reports(config, result)
    report_paths = write_agent_reports(agent_reports, out_dir) if out_dir else []

    out = {
        "transport": config.transport,
        "comparison": comparison_to_dict(comp) if comp is not None else None,
        "model_summaries": _model_summaries(result),
        "records": [vars(r) for r in result.records],
        "agent_reports": agent_reports,
        "report_paths": report_paths,
        "results_path": None,
        "summary": _summary_line(result, comp),
    }

    if out_dir:
        results_path = Path(out_dir) / "results.json"
        out["results_path"] = str(results_path)
        results_path.write_text(json.dumps(out, indent=2))

    return out
