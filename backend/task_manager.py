import asyncio
import uuid
# OPUS FIX: datetime.utcnow() is deprecated in 3.12+; use timezone-aware now(tz).
from datetime import datetime, timezone
from typing import Dict, Optional

from backend.schemas import TaskStatus, ResearchResponse


class TaskManager:
    """
    In-memory store for all research tasks.

    Lifecycle of a task:
      queued → running → complete | failed

    This is a simple dict-based store — suitable for a single-process server.
    For multi-process deployment, swap this for Redis.

    Designed as a singleton: one instance shared across all requests.
    """

    def __init__(self):
        self._tasks: Dict[str, ResearchResponse] = {}
        self._logs: Dict[str, list[str]] = {}   # task_id → log lines for WebSocket streaming
        self._lock = asyncio.Lock()

    async def create_task(self, query: str) -> str:
        """Creates a new task entry, returns the task_id."""
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        async with self._lock:
            self._tasks[task_id] = ResearchResponse(
                task_id=task_id,
                status=TaskStatus.queued,
                query=query,
                created_at=now,
                updated_at=now,
            )
            self._logs[task_id] = []

        return task_id

    async def set_running(self, task_id: str) -> None:
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.running
                self._tasks[task_id].updated_at = datetime.now(timezone.utc)

    async def set_complete(self, task_id: str, report: str,
                           sources: list[str], sections: list[str],
                           elapsed: float) -> None:
        async with self._lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.status = TaskStatus.complete
                task.report = report
                task.sources = sources
                task.sections = sections
                task.elapsed_seconds = elapsed
                task.updated_at = datetime.now(timezone.utc)

    async def set_failed(self, task_id: str, error: str) -> None:
        async with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.failed
                self._tasks[task_id].error = error
                self._tasks[task_id].updated_at = datetime.now(timezone.utc)

    async def get_task(self, task_id: str) -> Optional[ResearchResponse]:
        return self._tasks.get(task_id)

    async def list_tasks(self, limit: int = 20) -> list[ResearchResponse]:
        tasks = list(self._tasks.values())
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    async def add_log(self, task_id: str, message: str) -> None:
        """Appends a log line. Used by WebSocket streaming."""
        async with self._lock:
            if task_id in self._logs:
                timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
                self._logs[task_id].append(f"[{timestamp}] {message}")

    async def get_logs(self, task_id: str) -> list[str]:
        return self._logs.get(task_id, [])

    async def get_logs_since(self, task_id: str, since_index: int) -> list[str]:
        """Returns only new log lines since the given index. Used for WebSocket polling."""
        logs = self._logs.get(task_id, [])
        return logs[since_index:]

    async def delete_task(self, task_id: str) -> bool:
        """OPUS FIX: lock-protected deletion. Returns True if the task existed."""
        async with self._lock:
            existed = self._tasks.pop(task_id, None) is not None
            self._logs.pop(task_id, None)
            return existed


# Singleton instance — imported by all routes
task_manager = TaskManager()
