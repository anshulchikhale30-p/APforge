"""The REMEMBER step: persistent tool-use memory.

Success/failure counts are accumulated per (agent, task type, tool) in
SQLite. These statistics feed the strategy engine's tool selection, so
behavior learned on previous turns actually changes future tool choice.
"""

from __future__ import annotations

from typing import Optional

from .db import Database
from .models import ToolMemory, utcnow_iso


class ToolUseMemory:
    """Facade over the `tool_memory` table."""

    def __init__(self, db: Database):
        self.db = db

    def record(
        self,
        agent_id: str,
        task_type: str,
        tool_name: str,
        success: bool,
        used_at: Optional[str] = None,
    ) -> ToolMemory:
        """Record one tool use and return the updated memory row."""
        return self.db.upsert_memory(
            agent_id, task_type, tool_name, success=success, used_at=used_at
        )

    def get(self, agent_id: str, task_type: str, tool_name: str) -> Optional[ToolMemory]:
        return self.db.get_memory(agent_id, task_type, tool_name)

    def stats(
        self, agent_id: str, task_type: Optional[str] = None
    ) -> list[ToolMemory]:
        return self.db.list_memory(agent_id, task_type=task_type)

    def clear(self, agent_id: str) -> None:
        """Drop all memory rows for an agent (used by tests/ops)."""
        self.db.clear_memory(agent_id)

    @staticmethod
    def empty_used_at() -> str:
        return utcnow_iso()