from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    NEEDS_RETRY = "needs_retry"
    RESEARCHED = "researched"
    VERIFIED = "verified"
    DONE = "done"
    FAILED = "failed"


class Claim(BaseModel):
    """A single verifiable statement extracted from research notes.

    `source` is the evidence reference (URL or "kb:<chunk_id>") that the
    claim was drawn from / verified against. None means no supporting
    evidence was found — such claims must never receive a citation.
    """

    text: str
    source: str | None = None
    verified: bool = False


# Allowed status transitions. Anything else raises ValueError.
_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.IN_PROGRESS, TaskStatus.FAILED},
    TaskStatus.IN_PROGRESS: {TaskStatus.RESEARCHED, TaskStatus.NEEDS_RETRY, TaskStatus.FAILED},
    TaskStatus.NEEDS_RETRY: {TaskStatus.IN_PROGRESS, TaskStatus.FAILED},
    TaskStatus.RESEARCHED: {TaskStatus.VERIFIED, TaskStatus.NEEDS_RETRY, TaskStatus.FAILED},
    TaskStatus.VERIFIED: {TaskStatus.DONE, TaskStatus.FAILED},
    TaskStatus.DONE: set(),
    TaskStatus.FAILED: set(),
}


class Task(BaseModel):
    """A single unit of agent work tracked through the workflow."""

    id: str = Field(default_factory=lambda: uuid4().hex[:8])
    title: str = Field(min_length=3, max_length=200)
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: str = "researcher"
    evidence: list[str] = Field(default_factory=list)
    evidence_texts: list[str] = Field(default_factory=list)
    # Per-claim provenance. Parallel to evidence_texts via Claim.source refs.
    claims: list[Claim] = Field(default_factory=list)
    # Evidence ref -> display title (page title or chunk id), for citations.
    source_titles: dict[str, str] = Field(default_factory=dict)
    # Evidence ref -> heuristic credibility score in [0, 1] (see
    # agents/source_scorer.py). Populated by the researcher.
    evidence_credibility: dict[str, float] = Field(default_factory=dict)
    unverified_claims: list[str] = Field(default_factory=list)
    result: str | None = None
    retries: int = 0
    max_retries: int = 2
    tokens_in: int = 0
    tokens_out: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def transition(self, new_status: TaskStatus) -> "Task":
        allowed = _TRANSITIONS[self.status]
        if new_status not in allowed:
            raise ValueError(f"Illegal transition {self.status.value} -> {new_status.value}")
        self.status = new_status
        self.updated_at = datetime.now(timezone.utc)
        return self

    def can_retry(self) -> bool:
        return self.retries < self.max_retries

    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.DONE, TaskStatus.FAILED)
