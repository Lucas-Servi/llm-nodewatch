from __future__ import annotations

from typing import Protocol

from ..models import RunTrace


class StorageBackend(Protocol):
    """Protocol for run trace persistence."""

    def save(self, trace: RunTrace) -> None: ...
    def load(self, run_id: str) -> RunTrace | None: ...
    def list_runs(
        self,
        graph_name: str | None = None,
        since: str | None = None,
        conversation_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[RunTrace]: ...
    def delete(self, run_id: str) -> bool: ...
