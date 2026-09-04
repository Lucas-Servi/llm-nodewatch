"""
Production-ready nodewatch integration for a LangGraph chatbot service.

Demonstrates all 8 steps from the LangGraph Integration Guide in the README:
1. Feature-flagged import
2. Storage initialization (once per process)
3. Tracker creation (once per request)
4. Callback injection into LangGraph config
5. Finalize and persist trace
6. Access trace metrics
7. Error/timeout handling with always-finalize pattern
8. Cleanup on shutdown

Usage:
    python examples/langgraph_integration.py

Requirements:
    - pip install "llm-nodewatch[server]"
    - langgraph, langchain-core, langchain-openai installed
    - OPENAI_API_KEY set (or swap for any LangChain-compatible LLM)
"""

import os
import logging
import asyncio

logger = logging.getLogger(__name__)

# ─── Step 1: Feature-flagged import ─────────────────────────────────────────
# Makes nodewatch fully optional — your app works fine without it installed.
# Set NODEWATCH_ENABLED=0 to disable at runtime without uninstalling.

_NODEWATCH_AVAILABLE = False
_NODEWATCH_ENABLED = os.getenv("NODEWATCH_ENABLED", "1") == "1"
if _NODEWATCH_ENABLED:
    try:
        import nodewatch
        _NODEWATCH_AVAILABLE = True
    except ImportError:
        logger.debug("nodewatch not installed — tracing disabled")


class ChatbotService:
    """Example service wrapping a LangGraph agent with nodewatch observability."""

    def __init__(self):
        # ─── Step 2: Initialize storage once ────────────────────────────────
        # One SQLiteStorage per process lifetime. It uses WAL mode internally.
        self._nodewatch_storage = None
        if _NODEWATCH_AVAILABLE:
            try:
                db_path = os.getenv("NODEWATCH_DB", "./nodewatch.db")
                self._nodewatch_storage = nodewatch.SQLiteStorage(db_path)
            except Exception as exc:
                logger.warning("Nodewatch storage init failed: %r", exc)

    async def handle_query(
        self,
        graph,
        user_prompt: str,
        user_id: str = "",
        conversation_id: str = "",
    ) -> dict:
        """Process a user query through the graph with full observability."""

        # ─── Step 3: Create tracker per request ─────────────────────────────
        # One GraphTracker per graph.ainvoke() call.
        # - metadata: arbitrary key-value pairs stored with the trace
        # - storage + live=True: enables real-time monitoring via `nodewatch live`
        _tracker = None
        _trace = None
        if _NODEWATCH_AVAILABLE and self._nodewatch_storage is not None:
            try:
                _tracker = nodewatch.GraphTracker(
                    "my-agent",
                    metadata={
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                    },
                    storage=self._nodewatch_storage,
                    live=True,
                )
            except Exception as exc:
                logger.warning("Tracker creation failed: %r", exc)
                _tracker = None

        # ─── Step 4: Inject tracker as callback ─────────────────────────────
        # Merge into your existing config dict. The tracker is a standard
        # LangChain BaseCallbackHandler — no graph code changes needed.
        config = {
            "configurable": {"thread_id": conversation_id},
            "recursion_limit": 100,
        }
        if _tracker is not None:
            config["callbacks"] = [_tracker]

        # ─── Step 5 & 7: Invoke with error handling ─────────────────────────
        # CRITICAL: Always finalize even on exception/timeout.
        # Partial traces capture which nodes ran and where it stalled.
        try:
            result = await asyncio.wait_for(
                graph.ainvoke(
                    {"messages": [{"role": "user", "content": user_prompt}]},
                    config=config,
                ),
                timeout=600.0,
            )
            response_text = result["messages"][-1].content

            # Finalize on success
            if _tracker is not None:
                _trace = _tracker.finalize(
                    query=user_prompt, final_response=response_text
                )
                self._nodewatch_storage.save(_trace)

        except (TimeoutError, Exception) as exc:
            logger.exception("Graph execution failed: %r", exc)
            response_text = f"Error: {exc}"
            # Always finalize — partial traces help debug failures
            if _tracker is not None:
                try:
                    _trace = _tracker.finalize(query=user_prompt, final_response="")
                    self._nodewatch_storage.save(_trace)
                except Exception as trace_exc:
                    # Never let an observability failure mask the original error
                    # the caller is waiting on — but do say it happened.
                    logger.warning("could not persist partial trace: %s", trace_exc)

        # ─── Step 6: Access trace metrics ───────────────────────────────────
        trace_summary = None
        if _trace is not None:
            trace_summary = {
                "run_id": _trace.run_id,
                "total_tokens": _trace.total_tokens,
                "total_cost_usd": round(_trace.total_cost_usd, 6),
                "total_llm_calls": _trace.total_llm_calls,
                "total_tool_calls": _trace.total_tool_calls,
                "nodes_visited": _trace.nodes_visited,
                "duration_ms": round(_trace.total_duration_ms, 1),
            }

        return {"response": response_text, "trace": trace_summary}

    # ─── Step 8: Cleanup on shutdown ────────────────────────────────────────
    def shutdown(self):
        """Call this when your app shuts down (e.g., FastAPI lifespan)."""
        if self._nodewatch_storage is not None:
            try:
                self._nodewatch_storage.close()
            except Exception as exc:
                logger.warning("Error closing storage: %r", exc)


# ─── Demo: Build a minimal graph and run it ─────────────────────────────────

async def main():
    """Demonstrate the integration with a simple LangGraph agent."""
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from typing import Annotated
    from typing_extensions import TypedDict

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    llm = ChatOpenAI(model="gpt-4o-mini")

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()

    # Use the service
    service = ChatbotService()
    try:
        result = await service.handle_query(
            graph=graph,
            user_prompt="What is LangGraph?",
            user_id="demo-user",
            conversation_id="demo-conv-001",
        )
        print(f"Response: {result['response'][:200]}")
        if result["trace"]:
            print(f"Tokens: {result['trace']['total_tokens']}")
            print(f"Cost: ${result['trace']['total_cost_usd']}")
            print(f"Duration: {result['trace']['duration_ms']}ms")
            print(f"Run ID: {result['trace']['run_id']}")
    finally:
        service.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
