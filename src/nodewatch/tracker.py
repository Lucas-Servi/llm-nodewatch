"""Core callback handler that instruments LangGraph executions for per-node metrics."""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid as _uuid
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from .models import LLMCall, NodeSpan, RunTrace, ToolCall, classify_tool_error

# A truncated tool trace hides the bug you are tracing. Conversation 607's
# `search_metabolic_data` call carried 23 search terms and its answer ended in the
# omission notice that explained the whole failure — the 256/128 caps cut the terms
# mid-list and the notice off entirely, so the trace could not be diagnosed from
# nodewatch and needed the raw server log. Keep them generous but bounded; both are
# env-overridable for a deep-dive session.
TOOL_INPUT_CHARS = int(os.getenv("NODEWATCH_TOOL_INPUT_CHARS", "2000"))
TOOL_OUTPUT_PREVIEW_CHARS = int(os.getenv("NODEWATCH_TOOL_OUTPUT_CHARS", "1000"))


logger = logging.getLogger(__name__)


def _detect_provider(model: str) -> str:
    """Best-effort provider inference from a model identifier.

    Pure function (no side effects) so it can be unit-tested in isolation.
    Returns ``"unknown"`` when no known vendor token is present.
    """
    m = model.lower()
    if "claude" in m or "anthropic" in m:
        # Bedrock-served Claude ids carry a region routing prefix (us./eu./ap.).
        return "bedrock" if any(p in m for p in ("us.", "eu.", "ap.")) else "anthropic"
    if "gpt" in m or "o3" in m or "o4" in m:
        return "openai"
    if "gemini" in m or "google" in m:
        return "google"
    if "mistral" in m or "mixtral" in m or "codestral" in m:
        return "mistral"
    if "deepseek" in m:
        return "deepseek"
    if "command" in m or "cohere" in m:
        return "cohere"
    if "grok" in m or "xai" in m:
        return "xai"
    if "minimax" in m or "abab" in m:
        return "minimax"
    if "groq" in m:
        return "groq"
    if "llama" in m or "gemma" in m or "qwen" in m or "phi" in m:
        # Open-weight families commonly served via Groq or locally (Ollama).
        return "groq" if "groq" in m else "local"
    return "unknown"


def _usage_from_metadata(um: dict) -> dict[str, int]:
    """Normalize LangChain's ``usage_metadata`` into nodewatch's exclusive-input convention.

    LangChain reports ``input_tokens`` as the TOTAL prompt size (inclusive of cached tokens)
    and nests cache counts under ``input_token_details`` as ``cache_read`` / ``cache_creation``
    (verified for Anthropic, Bedrock-Converse and OpenAI). The rest of nodewatch (cost in
    models.py, cache-hit rate in stats.py) assumes ``input_tokens`` is EXCLUSIVE of cache, so we
    subtract the cache tokens back out here and surface them as separate fields.

    Reasoning tokens (``output_token_details.reasoning``) are intentionally NOT mapped to
    ``thinking_tokens``: providers already count reasoning inside ``output_tokens``, and
    nodewatch bills ``thinking_tokens`` separately at the output rate, so doing so would
    double-count them.
    """
    details = um.get("input_token_details") or {}
    cache_read = details.get("cache_read", 0) or 0
    cache_creation = details.get("cache_creation", 0) or 0
    total_input = um.get("input_tokens", 0) or 0
    return {
        "input_tokens": max(0, total_input - cache_read - cache_creation),
        "output_tokens": um.get("output_tokens", 0) or 0,
        "cache_read_input_tokens": cache_read,
        "cache_creation_input_tokens": cache_creation,
        "thinking_tokens": 0,
    }


def _usage_from_raw(raw: dict) -> dict[str, int]:
    """Normalize a raw provider usage dict (Anthropic/Bedrock/OpenAI SDK style).

    Here ``input_tokens`` already EXCLUDES cache tokens and cache counts use the raw API key
    names ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` — which is exactly the
    convention nodewatch stores, so no subtraction is needed.
    """
    return {
        "input_tokens": raw.get("input_tokens", 0) or raw.get("prompt_tokens", 0) or 0,
        "output_tokens": raw.get("output_tokens", 0) or raw.get("completion_tokens", 0) or 0,
        "cache_read_input_tokens": raw.get("cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": raw.get("cache_creation_input_tokens", 0) or 0,
        "thinking_tokens": raw.get("thinking_tokens", 0) or 0,
    }


class GraphTracker(BaseCallbackHandler):
    """Non-invasive LangChain callback handler that builds a RunTrace from LangGraph execution.

    Inject via config={"callbacks": [tracker]} — zero changes to graph code required.
    """

    def __init__(self, graph_name: str, metadata: dict | None = None, storage=None, live: bool = False, query: str = ""):
        self.graph_name = graph_name
        self.metadata = metadata or {}
        self._query = query
        self._start_time: float = time.time()
        self._lock = threading.Lock()
        self._run_id: str = _uuid.uuid4().hex[:12]

        # Live mode: persist intermediate state for real-time monitoring
        self._storage = storage
        self._live = live

        # Correlation state
        self._run_to_node: dict[UUID, str] = {}
        self._node_spans: dict[str, NodeSpan] = {}
        self._llm_starts: dict[UUID, tuple[float, str, str]] = {}  # run_id -> (start_time, node_name, model)
        self._tool_starts: dict[UUID, tuple[float, str, str, str]] = {}  # run_id -> (start_time, node_name, tool_name, input_str)

        # Track node iteration counts
        self._node_visit_counts: dict[str, int] = {}

        # Conversation tracking (extracted from LangGraph's thread_id)
        self._conversation_id: str = ""

        self._finalized = False
        self._error: str | None = None

        # Background heartbeat thread keeps `last_heartbeat` fresh during long operations
        self._heartbeat_stop = threading.Event()
        if self._live and self._storage:
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()
        else:
            self._heartbeat_thread = None

    @property
    def config(self) -> dict[str, Any]:
        """Convenience: returns a config dict ready to pass to graph.ainvoke()."""
        return {"callbacks": [self]}

    def _resolve_node(self, parent_run_id: UUID | None, tags: list[str] | None, metadata: dict | None) -> str:
        """Determine which graph node this event belongs to."""
        # LangGraph attaches a 'langgraph_node' key in metadata
        if metadata and "langgraph_node" in metadata:
            return metadata["langgraph_node"]

        # Fallback: check tags for node name patterns
        if tags:
            for tag in tags:
                if tag.startswith("graph:node:"):
                    return tag.split("graph:node:")[-1]

        # Fallback: inherit from parent run
        if parent_run_id and parent_run_id in self._run_to_node:
            return self._run_to_node[parent_run_id]

        return "__unknown__"

    def _get_or_create_span(self, node_name: str) -> NodeSpan:
        if node_name not in self._node_spans:
            self._node_spans[node_name] = NodeSpan(
                node_name=node_name,
                start_time=time.time(),
            )
        return self._node_spans[node_name]

    # ── Chain events (node entry/exit) ────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            # Extract thread_id (conversation_id) from LangGraph metadata
            if not self._conversation_id and metadata:
                thread_id = metadata.get("thread_id", "")
                if thread_id:
                    self._conversation_id = str(thread_id)

            node_name = self._resolve_node(parent_run_id, tags, metadata)
            self._run_to_node[run_id] = node_name

            if node_name != "__unknown__":
                # Only count as a new iteration if this is a top-level node entry
                # (parent resolves to a different node or to root)
                parent_node = self._run_to_node.get(parent_run_id) if parent_run_id else None
                is_top_level_entry = parent_node is None or parent_node != node_name

                span = self._get_or_create_span(node_name)
                if is_top_level_entry:
                    self._node_visit_counts[node_name] = self._node_visit_counts.get(node_name, 0) + 1
                    span.iterations = self._node_visit_counts[node_name]

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            node_name = self._run_to_node.get(run_id, "__unknown__")
            if node_name in self._node_spans:
                self._node_spans[node_name].end_time = time.time()
        self._maybe_save()

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            node_name = self._run_to_node.get(run_id, "__unknown__")
            self._error = f"[{node_name}] {type(error).__name__}: {error}"

    # ── LLM events ────────────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Extract model name from invocation_params or serialized kwargs
        invocation = kwargs.get("invocation_params") or {}
        model = (
            invocation.get("model")
            or invocation.get("model_id")
            or invocation.get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or serialized.get("kwargs", {}).get("model_id")
            or ""
        )
        with self._lock:
            node_name = self._resolve_node(parent_run_id, tags, metadata)
            self._run_to_node[run_id] = node_name
            self._llm_starts[run_id] = (time.time(), node_name, model)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            start_time, node_name, start_model = self._llm_starts.pop(run_id, (time.time(), "__unknown__", ""))
        duration_ms = (time.time() - start_time) * 1000

        # Extract token usage. Prefer LangChain's normalized usage_metadata (richest source,
        # carries cache + reasoning details); fall back to raw provider dicts. Both are passed
        # through _usage_from_* so input_tokens always lands in the exclusive-of-cache form
        # the rest of nodewatch expects.
        llm_output = response.llm_output or {}
        raw_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}

        usage: dict[str, int] = {}
        gen_info = {}
        if response.generations and response.generations[0]:
            gen = response.generations[0][0]
            if hasattr(gen, "generation_info") and gen.generation_info:
                gen_info = gen.generation_info
            message = getattr(gen, "message", None)
            um = getattr(message, "usage_metadata", None) or {}
            if um:
                usage = _usage_from_metadata(um)
            # Bedrock and some providers attach a raw usage dict under response_metadata
            if not usage:
                rm = getattr(message, "response_metadata", None) or {}
                rm_usage = rm.get("usage") or {}
                if rm_usage:
                    usage = _usage_from_raw(rm_usage)
        if not usage and raw_usage:
            usage = _usage_from_raw(raw_usage)

        # Detect model name: prefer what we captured at on_llm_start, then response
        model = start_model or llm_output.get("model_name", "") or llm_output.get("model", "")
        if not model and gen_info:
            model = gen_info.get("model", "")
        if not model and response.generations and response.generations[0]:
            gen = response.generations[0][0]
            if hasattr(gen, "message") and hasattr(gen.message, "response_metadata"):
                model = (gen.message.response_metadata or {}).get("model_id", "")

        # Detect provider from the model identifier (see _detect_provider).
        provider = _detect_provider(model)

        # Stop reason. Prefer generation_info, but fall back to the message's
        # response_metadata — ChatBedrockConverse exposes the stop reason only there,
        # under the camelCase key "stopReason" (never in generation_info). Without this
        # fallback every Bedrock call records an empty stop_reason, so content-filter
        # events (stopReason="content_filtered") would be invisible. Key order covers the
        # common provider conventions (Bedrock camelCase, then snake_case variants).
        stop_reason = ""
        if response.generations and response.generations[0]:
            gen = response.generations[0][0]
            if hasattr(gen, "generation_info") and gen.generation_info:
                stop_reason = gen.generation_info.get("stop_reason", "") or gen.generation_info.get("finish_reason", "")
            if not stop_reason:
                rm = getattr(getattr(gen, "message", None), "response_metadata", None) or {}
                stop_reason = (
                    rm.get("stopReason")
                    or rm.get("stop_reason")
                    or rm.get("finish_reason")
                    or rm.get("model_stop_reason")
                    or ""
                )

        llm_call = LLMCall(
            node_name=node_name,
            model=model,
            provider=provider,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            thinking_tokens=usage.get("thinking_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
            duration_ms=duration_ms,
            stop_reason=stop_reason,
        )

        with self._lock:
            span = self._get_or_create_span(node_name)
            span.llm_calls.append(llm_call)
        self._maybe_save()

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            start_time, node_name, start_model = self._llm_starts.pop(run_id, (time.time(), "__unknown__", ""))
        duration_ms = (time.time() - start_time) * 1000

        llm_call = LLMCall(
            node_name=node_name,
            model=start_model,
            provider="",
            duration_ms=duration_ms,
            error=f"{type(error).__name__}: {error}",
        )
        with self._lock:
            span = self._get_or_create_span(node_name)
            span.llm_calls.append(llm_call)

    # ── Tool events ───────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            node_name = self._resolve_node(parent_run_id, tags, metadata)
            self._run_to_node[run_id] = node_name
            tool_name = serialized.get("name", "") or serialized.get("id", ["", ""])[-1]
            self._tool_starts[run_id] = (time.time(), node_name, tool_name, input_str)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            start_time, node_name, tool_name, input_str = self._tool_starts.pop(run_id, (time.time(), "__unknown__", "", ""))
        duration_ms = (time.time() - start_time) * 1000

        output_str = str(output) if output else ""
        # Many tools return an error PAYLOAD instead of raising, so on_tool_error
        # never fires for them and this path would record success=True for a
        # genuine failure. Classify the FULL output, never the truncated preview.
        error = classify_tool_error(output_str)
        tool_call = ToolCall(
            node_name=node_name,
            tool_name=tool_name,
            duration_ms=duration_ms,
            success=error is None,
            error=error,
            input=input_str[:TOOL_INPUT_CHARS] if input_str else None,
            output_preview=output_str[:TOOL_OUTPUT_PREVIEW_CHARS] if output_str else None,
            output_size=len(output_str),
        )
        with self._lock:
            span = self._get_or_create_span(node_name)
            span.tool_calls.append(tool_call)
        self._maybe_save()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        with self._lock:
            start_time, node_name, tool_name, input_str = self._tool_starts.pop(run_id, (time.time(), "__unknown__", "", ""))
        duration_ms = (time.time() - start_time) * 1000

        tool_call = ToolCall(
            node_name=node_name,
            tool_name=tool_name,
            duration_ms=duration_ms,
            success=False,
            error=f"{type(error).__name__}: {error}",
            input=input_str[:TOOL_INPUT_CHARS] if input_str else None,
            output_size=0,
        )
        with self._lock:
            span = self._get_or_create_span(node_name)
            span.tool_calls.append(tool_call)

    # ── Live mode ─────────────────────────────────────────────────────────────

    def snapshot(self) -> RunTrace:
        """Create a point-in-time RunTrace from current state without finalizing."""
        with self._lock:
            spans = []
            for s in self._node_spans.values():
                if s.node_name == "__unknown__":
                    continue
                span_copy = NodeSpan(
                    node_name=s.node_name,
                    node_type=s.node_type,
                    start_time=s.start_time,
                    end_time=s.end_time,
                    llm_calls=list(s.llm_calls),
                    tool_calls=list(s.tool_calls),
                    iterations=s.iterations,
                )
                if span_copy.tool_calls and not span_copy.llm_calls:
                    span_copy.node_type = "tool_node"
                elif not span_copy.llm_calls and not span_copy.tool_calls:
                    span_copy.node_type = "router"
                else:
                    span_copy.node_type = "agent"
                spans.append(span_copy)

            return RunTrace(
                run_id=self._run_id,
                graph_name=self.graph_name,
                query=self._query,
                total_duration_ms=(time.time() - self._start_time) * 1000,
                node_spans=spans,
                error=self._error,
                metadata=self.metadata,
                conversation_id=self._conversation_id or self.metadata.get("conversation_id", ""),
            )

    def _heartbeat_loop(self):
        """Background thread that saves snapshots every 30s to keep heartbeat fresh."""
        while not self._heartbeat_stop.wait(30.0):
            if self._finalized:
                break
            try:
                trace = self.snapshot()
                self._storage.save(trace, status="running")
            except Exception as e:
                logger.debug("heartbeat save failed: %s", e)

    def _maybe_save(self):
        """Persist a snapshot if live mode is enabled."""
        if self._live and self._storage:
            try:
                trace = self.snapshot()
                self._storage.save(trace, status="running")
            except Exception as e:
                logger.debug("live snapshot failed: %s", e)

    # ── Async wrappers (eliminate fallback hop for async LangGraph graphs) ───

    async def aon_chain_start(self, *args: Any, **kwargs: Any) -> None:
        self.on_chain_start(*args, **kwargs)

    async def aon_chain_end(self, *args: Any, **kwargs: Any) -> None:
        self.on_chain_end(*args, **kwargs)

    async def aon_chain_error(self, *args: Any, **kwargs: Any) -> None:
        self.on_chain_error(*args, **kwargs)

    async def aon_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self.on_llm_start(*args, **kwargs)

    async def aon_llm_end(self, *args: Any, **kwargs: Any) -> None:
        self.on_llm_end(*args, **kwargs)

    async def aon_llm_error(self, *args: Any, **kwargs: Any) -> None:
        self.on_llm_error(*args, **kwargs)

    async def aon_tool_start(self, *args: Any, **kwargs: Any) -> None:
        self.on_tool_start(*args, **kwargs)

    async def aon_tool_end(self, *args: Any, **kwargs: Any) -> None:
        self.on_tool_end(*args, **kwargs)

    async def aon_tool_error(self, *args: Any, **kwargs: Any) -> None:
        self.on_tool_error(*args, **kwargs)

    # ── Finalization ──────────────────────────────────────────────────────────

    def finalize(self, query: str = "", final_response: str = "") -> RunTrace:
        """Assemble all captured events into a complete RunTrace."""
        if self._finalized:
            raise RuntimeError("GraphTracker.finalize() already called")
        self._finalized = True

        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2.0)

        end_time = time.time()
        total_duration_ms = (end_time - self._start_time) * 1000

        # Collect spans, filtering out the unknown bucket
        spans = [s for s in self._node_spans.values() if s.node_name != "__unknown__"]

        # Classify node types based on contents
        for span in spans:
            if span.tool_calls and not span.llm_calls:
                span.node_type = "tool_node"
            elif not span.llm_calls and not span.tool_calls:
                span.node_type = "router"
            else:
                span.node_type = "agent"

        trace = RunTrace(
            run_id=self._run_id,
            graph_name=self.graph_name,
            query=query or self._query,
            total_duration_ms=total_duration_ms,
            node_spans=spans,
            final_response=final_response,
            error=self._error,
            metadata=self.metadata,
            conversation_id=self._conversation_id or self.metadata.get("conversation_id", ""),
        )

        if self._live and self._storage:
            self._storage.save(trace, status="done")

        return trace
