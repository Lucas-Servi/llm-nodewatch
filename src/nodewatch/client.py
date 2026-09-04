"""Remote HTTP client for the nodewatch API.

When NODEWATCH_URL is set, the CLI uses this instead of local SQLite access.

Usage (direct server access, no auth):
    export NODEWATCH_URL=http://server:8000/nodewatch/
    nodewatch report --last 5

Usage (with token authentication):
    export NODEWATCH_URL=https://your-api.example.com/api/v1/nodewatch
    export NODEWATCH_TOKEN=your_access_token
    nodewatch report --last 5

Usage (with auto-login via custom module):
    export NODEWATCH_URL=https://your-api.example.com/api/v1/nodewatch
    export NODEWATCH_LOGIN_MODULE=mypackage.auth
    export NODEWATCH_LOGIN_FUNCTION=get_token
    nodewatch report --last 5
"""

from __future__ import annotations

import json as _json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests
except ImportError:
    requests = None  # type: ignore[assignment]

from .models import LLMCall, NodeSpan, RunTrace, ToolCall  # noqa: E402


def _serialize_input(val: Any) -> str | None:
    """Convert input field (may be dict, list, or str) to a JSON string."""
    if val is None:
        return None
    if isinstance(val, str):
        return val
    return _json.dumps(val)


def get_remote_url() -> str | None:
    """Return the remote API URL if configured, None otherwise."""
    return os.getenv("NODEWATCH_URL")


def _get_token() -> str | None:
    """Obtain an access token for authenticated API calls.

    Resolution order:
    1. NODEWATCH_TOKEN env var (pre-obtained token)
    2. NODEWATCH_LOGIN_MODULE + NODEWATCH_LOGIN_FUNCTION (custom login callable)
    3. None (no authentication)
    """
    token = os.getenv("NODEWATCH_TOKEN")
    if token:
        return token

    login_module = os.getenv("NODEWATCH_LOGIN_MODULE")
    login_function = os.getenv("NODEWATCH_LOGIN_FUNCTION", "login")
    if login_module:
        import importlib
        mod = importlib.import_module(login_module)
        fn = getattr(mod, login_function)
        if not callable(fn):
            raise RuntimeError(f"NODEWATCH_LOGIN_FUNCTION '{login_module}.{login_function}' is not callable")
        token = fn()
        if not isinstance(token, str):
            raise RuntimeError(f"NODEWATCH_LOGIN_FUNCTION '{login_module}.{login_function}' must return a str, got {type(token).__name__}")
        return token

    return None


@dataclass
class RemoteClient:
    """HTTP client that mirrors the SQLiteStorage interface for CLI use.

    Supports two modes:
    - Direct: NODEWATCH_URL points to the /nodewatch/ prefix on the internal server
    - Public API: NODEWATCH_URL points to the public API base, authentication via login()
    """

    base_url: str
    _token: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if requests is None:
            raise ImportError("'requests' package required for remote mode. Install with: pip install requests")
        self.base_url = self.base_url.rstrip("/")
        self._token = _get_token()

    def _params(self) -> dict:
        """Query params with access_token if authenticated."""
        if self._token:
            return {"access_token": self._token}
        return {}

    def _dispatch_url(self) -> str:
        """URL for the dispatch endpoint."""
        if "nodewatch-dispatch" not in self.base_url and "/nodewatch" not in self.base_url:
            return f"{self.base_url}/nodewatch-dispatch"
        if self.base_url.endswith("/nodewatch"):
            return f"{self.base_url}/"
        return self.base_url

    def _methods_url(self) -> str:
        """URL for the methods discovery endpoint."""
        if "nodewatch-get-methods" not in self.base_url and "/nodewatch" not in self.base_url:
            return f"{self.base_url}/nodewatch-get-methods"
        if self.base_url.endswith("/nodewatch"):
            return f"{self.base_url}/methods"
        return f"{self.base_url.rsplit('/', 1)[0]}/methods"

    def _call(self, method: str, _retried: bool = False, **args) -> Any:
        """Call a remote API method. Retries once on auth errors."""
        payload = {"method": method, "args": {k: v for k, v in args.items() if v is not None}}
        try:
            resp = requests.post(self._dispatch_url(), json=payload, params=self._params(), timeout=30)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Cannot reach server ({self.base_url}): {e}") from e
        except requests.exceptions.Timeout as e:
            raise ConnectionError(f"Server timed out ({self.base_url})") from e
        if resp.status_code in (400, 422):
            logger.debug("Remote API error (%d): %s", resp.status_code, resp.text)
            raise RuntimeError(f"Remote API error ({resp.status_code}); set NODEWATCH_DEBUG=1 for details")
        resp.raise_for_status()
        body = resp.json()
        if "data" in body:
            body = body["data"]
        if isinstance(body, dict) and body.get("error"):
            msg = body.get("message", "unknown error")
            if not _retried:
                logger.debug("Retrying after error (possible token expiry): %s", msg)
                self._token = _get_token()
                return self._call(method, _retried=True, **args)
            logger.debug("Remote API returned error after retry: %s", msg)
            raise RuntimeError(f"Remote API error: {msg}")
        return body.get("result", body)

    def list_runs(self, graph_name: str | None = None, since: str | None = None, conversation_id: str | None = None, limit: int = 50, offset: int = 0) -> list[RunTrace]:
        """Fetch run summaries and convert to RunTrace objects."""
        data = self._call("list_runs", graph_name=graph_name, since=since, conversation_id=conversation_id, limit=limit, offset=offset)
        return [_summary_to_trace(d) for d in data]

    def load(self, run_id: str) -> RunTrace | None:
        """Fetch a full trace by ID."""
        try:
            data = self._call("get_run", run_id=run_id)
        except RuntimeError:
            return None
        return _json_to_trace(data)

    def delete(self, run_id: str) -> bool:
        """Delete a trace by ID."""
        try:
            self._call("delete_run", run_id=run_id)
            return True
        except RuntimeError:
            return False

    def report(self, graph_name: str | None = None, since: str | None = None, limit: int = 50) -> dict:
        """Fetch pre-computed aggregate stats."""
        return self._call("report", graph_name=graph_name, since=since, limit=limit)

    def ab_compare(self, expected_a: str = "opus-4-8", expected_b: str = "opus-4-7",
                   since: str | None = None, limit: int = 200) -> dict:
        """Fetch a server-computed A/B model comparison (verification + per-question deltas)."""
        return self._call("ab_compare", expected_a=expected_a, expected_b=expected_b,
                          since=since, limit=limit)

    def get_methods(self) -> dict:
        """Fetch available API methods."""
        try:
            resp = requests.get(self._methods_url(), params=self._params(), timeout=10)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            raise ConnectionError(f"Cannot reach server ({self.base_url})") from e
        resp.raise_for_status()
        body = resp.json()
        if "data" in body:
            body = body["data"]
        return body.get("methods", {})

    def list_active_runs(self, stale_timeout_s: float = 120.0) -> list[dict]:
        """Fetch currently running traces."""
        try:
            result = self._call("get_active_runs")
            return result if isinstance(result, list) else []
        except (RuntimeError, Exception):
            return []

    def load_live(self, run_id: str) -> dict | None:
        """Fetch a running trace with status metadata."""
        try:
            return self._call("get_run_live", run_id=run_id)
        except RuntimeError:
            return None

    def get_status(self, run_id: str) -> str | None:
        """Get status from a live trace response."""
        data = self.load_live(run_id)
        return data.get("status") if data else None

    def get_logs(self, position: int = -1) -> dict:
        """Fetch log lines from server."""
        try:
            return self._call("get_logs", position=position)
        except (RuntimeError, Exception):
            return {"lines": [], "position": 0}

    def close(self):
        pass


def _summary_to_trace(data: dict) -> RunTrace:
    """Convert a list_runs summary dict to a minimal RunTrace."""
    ts = data.get("timestamp", "")
    try:
        timestamp = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        timestamp = datetime.now(timezone.utc)

    return RunTrace(
        run_id=data.get("run_id", ""),
        graph_name=data.get("graph_name", ""),
        query=data.get("query", ""),
        timestamp=timestamp,
        total_duration_ms=data.get("duration_ms", 0.0),
        node_spans=[],
        error=data.get("error"),
        metadata={},
        conversation_id=data.get("conversation_id", ""),
        _tokens_override=data.get("total_tokens", 0),
        _cost_override=data.get("total_cost_usd", 0.0),
    )


def _json_to_trace(data: dict) -> RunTrace:
    """Convert a full trace JSON dict to a RunTrace object."""
    ts = data.get("timestamp", "")
    try:
        timestamp = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        timestamp = datetime.now(timezone.utc)

    node_spans = []
    for ns in data.get("node_spans", []):
        llm_calls = [
            LLMCall(
                node_name=ns.get("node_name", ""),
                model=lc.get("model", ""),
                provider=lc.get("provider", ""),
                input_tokens=lc.get("input_tokens", 0),
                output_tokens=lc.get("output_tokens", 0),
                thinking_tokens=lc.get("thinking_tokens", 0),
                cache_read_tokens=lc.get("cache_read_tokens", 0),
                cache_creation_tokens=lc.get("cache_creation_tokens", 0),
                duration_ms=lc.get("duration_ms", 0.0),
                stop_reason=lc.get("stop_reason", ""),
                error=lc.get("error"),
            )
            for lc in ns.get("llm_calls", [])
        ]
        tool_calls = [
            ToolCall(
                node_name=ns.get("node_name", ""),
                tool_name=tc.get("tool_name", ""),
                duration_ms=tc.get("duration_ms", 0.0),
                success=tc.get("success", True),
                error=tc.get("error"),
                input=_serialize_input(tc.get("input")),
                output_preview=tc.get("output_preview"),
                output_size=tc.get("output_size", 0),
            )
            for tc in ns.get("tool_calls", [])
        ]
        raw_start = ns.get("start_time", 0.0) or 0.0
        raw_end = ns.get("end_time", 0.0) or 0.0
        raw_dur_ms = ns.get("duration_ms", 0.0) or 0.0
        # If no absolute times but duration exists, synthesize relative times
        if raw_start == 0.0 and raw_end == 0.0 and raw_dur_ms > 0:
            raw_start = 1.0  # synthetic non-zero start
            raw_end = 1.0 + raw_dur_ms / 1000
        node_spans.append(NodeSpan(
            node_name=ns.get("node_name", ""),
            node_type=ns.get("node_type", "agent"),
            start_time=raw_start,
            end_time=raw_end,
            llm_calls=llm_calls,
            tool_calls=tool_calls,
            iterations=ns.get("iterations", 0),
        ))

    return RunTrace(
        run_id=data.get("run_id", ""),
        graph_name=data.get("graph_name", ""),
        query=data.get("query", ""),
        timestamp=timestamp,
        total_duration_ms=data.get("total_duration_ms", 0.0),
        node_spans=node_spans,
        final_response=data.get("final_response", ""),
        error=data.get("error"),
        metadata=data.get("metadata", {}),
        conversation_id=data.get("conversation_id", ""),
    )
