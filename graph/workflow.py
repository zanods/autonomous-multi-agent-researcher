from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from agents.fact_checker import verify_task
from agents.planner import plan_tasks
from agents.researcher import research_task
from agents.synthesizer import synthesize
from graph.state import AgentState
from models.task import TaskStatus
from optimization.budget import TokenBudget


def _log(state: AgentState, msg: str) -> list[str]:
    return [*state.get("run_log", []), msg]


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
    """Compile the Sprint 1 graph: planner -> researcher -> fact-checker -> synthesizer."""
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("fact_checker", fact_checker_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "fact_checker")
    graph.add_conditional_edges("fact_checker", route_after_check)
    graph.add_edge("synthesizer", END)
    return graph.compile()
