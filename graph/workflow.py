from __future__ import annotations

import asyncio

from langgraph.graph import StateGraph, START, END

from agents.fact_checker import verify_task
from agents.planner import plan_tasks
from agents.researcher import research_task
from agents.synthesizer import synthesize
from graph.state import AgentState
from models.task import Task, TaskStatus
from optimization.budget import TokenBudget


def _log(state: AgentState, msg: str) -> list[str]:
    return [*state.get("run_log", []), msg]


def run_task_pipeline(
    task: Task, mock_mode: bool = True, budget: TokenBudget | None = None
) -> Task:
    """Run one task through research -> verify with retries.

    Blocking and thread-safe: touches only this task (plus its own budget).
    Follows the same status machine and retry cap (max_retries) as the
    sequential graph nodes. research_task/verify_task are resolved as module
    globals so tests can patch them.
    """
    budget = budget or TokenBudget()
    task.transition(TaskStatus.IN_PROGRESS)
    while True:
        research_task(
            task,
            mock_mode=mock_mode,
            budget=budget,
            focus_claims=task.unverified_claims or None,
        )
        task.transition(TaskStatus.RESEARCHED)
        ok, _note, failed = verify_task(task, mock_mode=mock_mode, budget=budget)
        if ok:
            task.unverified_claims = []
            task.transition(TaskStatus.VERIFIED)
            task.transition(TaskStatus.DONE)
            return task
        if task.can_retry():
            task.retries += 1
            task.unverified_claims = failed
            task.transition(TaskStatus.NEEDS_RETRY)
            task.transition(TaskStatus.IN_PROGRESS)
            continue
        task.transition(TaskStatus.FAILED)
        return task


async def arun_tasks(
    tasks: list[Task], mock_mode: bool = True
) -> list[tuple[Task, TokenBudget]]:
    """Run independent tasks concurrently.

    LLM/HTTP calls are I/O-bound, so each task runs in a worker thread via
    asyncio.to_thread. Every task gets its own TokenBudget (no shared
    mutation); per-task retry logic is preserved by run_task_pipeline.
    """

    async def _one(task: Task) -> tuple[Task, TokenBudget]:
        budget = TokenBudget()
        result = await asyncio.to_thread(run_task_pipeline, task, mock_mode, budget)
        return result, budget

    return await asyncio.gather(*[_one(t) for t in tasks])


def _merge_budget(into: TokenBudget, other: TokenBudget) -> None:
    for node, n in other.usage.items():
        into.usage[node] = into.usage.get(node, 0) + n
    for node, m in other.measured.items():
        slot = into.measured.setdefault(node, {"input": 0, "output": 0})
        slot["input"] += m["input"]
        slot["output"] += m["output"]


def research_all_node(state: AgentState) -> dict:
    """Concurrent research phase: all pending tasks run at once.

    Emits one run_log line per completed task (task granularity, not
    per-step spam) so the CLI/dashboard can stream progress.
    """
    tasks = [
        t
        for t in state["tasks"]
        if t.status in (TaskStatus.PENDING, TaskStatus.NEEDS_RETRY)
    ]
    budget = state.get("token_budget") or TokenBudget()
    results = asyncio.run(
        arun_tasks(tasks, mock_mode=state.get("mock_mode", True))
    )
    logs = state.get("run_log", [])
    for task, b in results:
        _merge_budget(budget, b)
        logs.append(f"[researcher] {task.status.value}: {task.title}")
    return {"tasks": state["tasks"], "token_budget": budget, "run_log": logs}


def planner_node(state: AgentState) -> dict:
    budget = state.get("token_budget") or TokenBudget()
    tasks = plan_tasks(
        state["query"], mock_mode=state.get("mock_mode", True), budget=budget
    )
    return {
        "tasks": tasks,
        "token_budget": budget,
        "run_log": _log(state, f"[planner] created {len(tasks)} tasks"),
    }


def researcher_node(state: AgentState) -> dict:
    tasks = state["tasks"]
    nxt = next(
        (t for t in tasks if t.status in (TaskStatus.PENDING, TaskStatus.NEEDS_RETRY)),
        None,
    )
    if nxt is None:
        return {"run_log": _log(state, "[researcher] nothing to do")}
    budget = state.get("token_budget") or TokenBudget()
    nxt.transition(TaskStatus.IN_PROGRESS)
    # On retry, narrow research to the claims that failed verification.
    focus = nxt.unverified_claims or None
    research_task(nxt, mock_mode=state.get("mock_mode", True), budget=budget,
                  focus_claims=focus)
    nxt.transition(TaskStatus.RESEARCHED)
    return {
        "tasks": tasks,
        "current_task_id": nxt.id,
        "token_budget": budget,
        "run_log": _log(state, f"[researcher] researched: {nxt.title}"),
    }


def fact_checker_node(state: AgentState) -> dict:
    tasks = state["tasks"]
    cur = next(t for t in tasks if t.id == state.get("current_task_id"))
    ok, note, failed = verify_task(
        cur, mock_mode=state.get("mock_mode", True), budget=state.get("token_budget")
    )
    if ok:
        cur.unverified_claims = []
        cur.transition(TaskStatus.VERIFIED)
        cur.transition(TaskStatus.DONE)
        msg = f"[fact-checker] verified: {cur.title}"
    elif cur.can_retry():
        cur.retries += 1
        cur.unverified_claims = failed
        cur.transition(TaskStatus.NEEDS_RETRY)
        msg = f"[fact-checker] weak evidence, retry {cur.retries}/{cur.max_retries}: {cur.title} ({note})"
    else:
        cur.transition(TaskStatus.FAILED)
        msg = f"[fact-checker] failed after retries: {cur.title} ({note})"
    return {"tasks": tasks, "run_log": _log(state, msg)}


def route_after_check(state: AgentState) -> str:
    remaining = [
        t for t in state["tasks"]
        if t.status in (TaskStatus.PENDING, TaskStatus.NEEDS_RETRY)
    ]
    return "researcher" if remaining else "synthesizer"


def synthesizer_node(state: AgentState) -> dict:
    budget = state.get("token_budget")
    report = synthesize(
        state["query"],
        state["tasks"],
        mock_mode=state.get("mock_mode", True),
        budget=budget,
    )
    msg = "[synthesizer] report ready"
    if budget is not None and budget.measured:
        msg += f" | tokens used: {budget.total_measured()}"
    elif budget is not None and budget.usage:
        msg += f" | context tokens used (est.): {budget.total_tokens()}"
    return {"report": report, "run_log": _log(state, msg)}


def build_workflow():
    """Compile the graph: planner -> research_all (concurrent) -> synthesizer.

    The legacy sequential nodes (researcher_node / fact_checker_node /
    route_after_check) are kept for unit tests but are no longer wired in.
    """
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("research_all", research_all_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "research_all")
    graph.add_edge("research_all", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph.compile()
