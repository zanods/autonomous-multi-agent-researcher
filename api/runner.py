from __future__ import annotations

from agents.fact_checker import verify_task
from agents.llm import has_api_key
from agents.researcher import research_task
from agents.synthesizer import synthesize
from api import store
from models.task import Task, TaskStatus
from optimization.budget import TokenBudget


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


def build_report(goal_id: str) -> str:
    goal = store.get_goal(goal_id)
    if goal is None:
        raise KeyError(goal_id)
    tasks = [Task(**r) for r in store.list_tasks(goal_id)]
    return synthesize(goal["query"], tasks, mock_mode=_mock_mode())
