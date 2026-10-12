from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone

from agents.fact_checker import verify_task
from agents.llm import has_api_key
from agents.researcher import research_task
from agents.synthesizer import synthesize
from api import store
from models.task import Task, TaskStatus
from optimization.budget import TokenBudget

# Per-task completion events (task granularity, not per-step spam) for the
# dashboard to poll. Bounded; in-process only.
TASK_EVENTS: deque = deque(maxlen=1000)


def _emit_event(task_id: str, goal_id: str | None, status: str) -> None:
    TASK_EVENTS.append(
        {
            "task_id": task_id,
            "goal_id": goal_id,
            "status": status,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    )


def get_events(goal_id: str | None = None) -> list[dict]:
    return [
        e for e in TASK_EVENTS if goal_id is None or e["goal_id"] == goal_id
    ]


def clear_events() -> None:
    TASK_EVENTS.clear()


def _mock_mode() -> bool:
    return not has_api_key()


def _record_tokens(task: Task, budget: TokenBudget) -> None:
    for m in budget.measured.values():
        task.tokens_in += m["input"]
        task.tokens_out += m["output"]


def run_task(task_id: str) -> None:
    """Background job: research + fact-check one task, persisting each step.

    Safe to call twice: claim_task() atomically claims PENDING / NEEDS_RETRY
    tasks, so a concurrent second call is a no-op. Mirrors the workflow's
    retry logic — on retry, research narrows to the claims that failed
    verification instead of starting over.
    """
    row = store.claim_task(task_id)
    if row is None:
        return
    task = Task(**row)

    mock = _mock_mode()
    budget = TokenBudget()
    # On retry, narrow research to the claims that failed verification.
    focus = task.unverified_claims or None
    try:
        research_task(task, mock_mode=mock, budget=budget, focus_claims=focus)
        task.transition(TaskStatus.RESEARCHED)
        store.save_task(task)

        ok, _note, failed = verify_task(task, mock_mode=mock, budget=budget)
        if ok:
            task.unverified_claims = []
            task.transition(TaskStatus.VERIFIED)
            task.transition(TaskStatus.DONE)
        elif task.can_retry():
            task.retries += 1
            task.unverified_claims = failed
            task.transition(TaskStatus.NEEDS_RETRY)
        else:
            task.transition(TaskStatus.FAILED)
    except Exception as e:
        # Keep the failure cause — a bare FAILED with no reason is undebuggable.
        task.result = f"error: {type(e).__name__}: {e}"
        task.transition(TaskStatus.FAILED)
    _record_tokens(task, budget)
    store.save_task(task)
    if task.is_terminal():
        _emit_event(task_id, row.get("goal_id"), task.status.value)


async def _arun_task_ids(task_ids: list[str], max_workers: int) -> None:
    sem = asyncio.Semaphore(max_workers)

    async def _one(tid: str) -> None:
        async with sem:
            # run_task is blocking I/O (LLM HTTP calls); thread it.
            await asyncio.to_thread(run_task, tid)

    await asyncio.gather(*[_one(tid) for tid in task_ids])


def run_all_tasks(
    goal_id: str, max_workers: int = 4, max_rounds: int = 3
) -> list[str]:
    """Run every claimable task of a goal concurrently.

    Each round runs the currently pending/needs_retry tasks at once
    (asyncio + worker threads); rounds repeat so NEEDS_RETRY tasks get
    their retries, bounded by max_rounds. Per-task retry caps still apply
    inside run_task, and claim_task() stays a single atomic UPDATE, so
    double-queueing the same goal is safe: losers of the claim race no-op.
    """
    queued: list[str] = []
    for _ in range(max_rounds):
        tids = [
            t["id"]
            for t in store.list_tasks(goal_id)
            if t["status"] in ("pending", "needs_retry")
        ]
        if not tids:
            break
        queued.extend(tids)
        asyncio.run(_arun_task_ids(tids, max_workers))
    return queued


def build_report(goal_id: str) -> str:
    goal = store.get_goal(goal_id)
    if goal is None:
        raise KeyError(goal_id)
    tasks = [Task(**r) for r in store.list_tasks(goal_id)]
    return synthesize(goal["query"], tasks, mock_mode=_mock_mode())
