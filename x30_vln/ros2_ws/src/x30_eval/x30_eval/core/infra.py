"""Infrastructure failures: problems of the test bench, not of the navigation system under test."""
from __future__ import annotations

import re

from .recording import TaskResult


class InfraError(RuntimeError):
    """The environment cannot run the episode (service missing, sim crashed, container down, robot never ready)."""

    def __init__(self, message: str, kind: str = "infrastructure"):
        super().__init__(message)
        self.kind = kind


def looks_like_infra(task: TaskResult, patterns: list[str]) -> str | None:
    """Return the matching pattern when the task result looks like a bench failure (connection errors etc.).

    Only results that did not finish normally are inspected: a successful or timed-out run is never reclassified."""
    if task.stop_reason in ("finished", "timeout", "evaluator_timeout"):
        return None
    text = f"{task.stop_reason} {task.message}"
    for pat in patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            return pat
    return None
