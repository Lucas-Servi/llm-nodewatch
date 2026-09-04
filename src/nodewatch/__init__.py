"""llm-nodewatch: LangGraph-specific agent benchmarking and observability."""

from .abrun import (
    ABRunConfig,
    ApiConfig,
    ModelConfig,
    init_session,
    load_ab_config,
    preview_ab_config,
    resolve_session_dir,
    run_ab_config,
)
from .charts import render_bar_chart, render_summary
from .client import RemoteClient, get_remote_url
from .experiment import (
    ABExperiment,
    ExperimentQuestion,
    ExperimentResult,
    ExperimentSpec,
    Phase,
    PhaseRunRecord,
)
from .inspector import inspect_graph
from .models import (
    LLMCall,
    NodeSpan,
    RunTrace,
    ToolCall,
    classify_tool_error,
    is_filtered_stop,
)
from .reporter import (
    ab_comparison_to_markdown,
    comparison_to_markdown,
    trace_to_json,
    trace_to_markdown,
)
from .runner import BenchmarkRunner, ComparisonReport, Query
from .stats import (
    ABComparison,
    CohortVerification,
    QuestionDelta,
    SummaryStats,
    compute_ab_comparison,
    compute_summary,
    extract_chart_data,
    node_sig,
)
from .storage import SQLiteStorage
from .tracker import GraphTracker

try:
    from .api import create_router
except ImportError:
    create_router = None  # type: ignore[assignment,misc]

__all__ = [
    "GraphTracker",
    "BenchmarkRunner",
    "Query",
    "ComparisonReport",
    "RunTrace",
    "NodeSpan",
    "LLMCall",
    "ToolCall",
    "is_filtered_stop",
    "classify_tool_error",
    "SQLiteStorage",
    "trace_to_markdown",
    "comparison_to_markdown",
    "ab_comparison_to_markdown",
    "trace_to_json",
    "compute_summary",
    "compute_ab_comparison",
    "node_sig",
    "ABComparison",
    "CohortVerification",
    "QuestionDelta",
    "extract_chart_data",
    "SummaryStats",
    "ABExperiment",
    "ExperimentSpec",
    "Phase",
    "ExperimentQuestion",
    "PhaseRunRecord",
    "ExperimentResult",
    "load_ab_config",
    "run_ab_config",
    "preview_ab_config",
    "init_session",
    "resolve_session_dir",
    "ABRunConfig",
    "ApiConfig",
    "ModelConfig",
    "render_bar_chart",
    "render_summary",
    "RemoteClient",
    "get_remote_url",
    "inspect_graph",
    "create_router",
]

__version__ = "0.2.0"  # keep in sync with pyproject.toml (tests/test_version.py enforces it)
