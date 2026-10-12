"""Programmatic eval runner: drives the pipeline with mocked LLMs (no quota
burn) and computes three metrics:

  trap_catch_rate        % of trap scenarios whose tasks correctly FAILED
                         after retries (fabricated claims never verified)
  verification_precision % of "verified" claims that carry a cited source
  retry_stability_rate   % of scenarios where per-retry claim counts never
                         grow (no 27->34 style bloat)

Runnable:  python -m eval.run_eval
"""

from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from unittest.mock import patch

from eval.trap_queries import SCENARIOS
from models.task import Task, TaskStatus
from optimization.claims import split_claims


@dataclass
class ScenarioResult:
    scenario_id: str
    kind: str
    tasks_total: int
    tasks_done: int
    tasks_failed: int
    trap_caught: bool | None
    verified_claims: int
    verified_with_source: int
    retry_stable: bool
    attempts: list[list[int]] = field(default_factory=list)  # claim counts per task


@dataclass
class EvalSummary:
    trap_catch_rate: float | None
    verification_precision: float | None
    retry_stability_rate: float
    scenarios: list[ScenarioResult]


class _Resp:
    def __init__(self, content: str):
        self.content = content


class _ResearcherLLM:
    """Returns the scenario's canned notes on every attempt (incl. retries)."""

    def __init__(self, notes: str):
        self.notes = notes

    def invoke(self, prompt):
        return _Resp(self.notes)


class _CheckerLLM:
    """Emits one verdict line per numbered claim found in the prompt."""

    def __init__(self, mode: str):
        assert mode in ("fail_all", "pass_all")
        self.mode = mode

    def invoke(self, prompt):
        n = len(re.findall(r"(?m)^\d+\.\s", prompt))
        verdict = "FAIL" if self.mode == "fail_all" else "PASS"
        return _Resp("\n".join(f"{i + 1}: {verdict}" for i in range(max(n, 1))))


class _SynthLLM:
    def invoke(self, prompt):
        return _Resp("Eval report (mocked).")


@contextmanager
def _patched_pipeline(scenario: dict, attempts: list):
    """Swap the LLM/retrieval layer for deterministic fakes. `attempts`
    records (task_id, claim_count) on every fact-checker invocation."""
    import agents.fact_checker as fc_mod
    import agents.researcher as rs_mod
    import agents.synthesizer as sy_mod
    import graph.workflow as wf_mod

    tasks = [
        Task(title=title, description=desc) for title, desc in scenario["tasks"]
    ]
    hits = [
        {"title": title, "snippet": text, "url": ref}
        for ref, title, text in scenario.get("evidence", [])
    ]
    real_verify = fc_mod.verify_task

    def spy_verify(task, **kwargs):
        n = len(task.claims) if task.claims else len(split_claims(task.result or ""))
        out = real_verify(task, **kwargs)
        attempts.append((task.id, n))
        return out

    with (
        patch.object(
            wf_mod,
            "plan_tasks",
            lambda query, mock_mode=False, budget=None: tasks,
        ),
        patch.object(rs_mod, "get_llm", lambda: _ResearcherLLM(scenario["notes"])),
        patch.object(rs_mod, "rag_search", lambda q, k=4: []),
        patch.object(rs_mod, "web_search", lambda q, max_results=5: hits),
        patch.object(fc_mod, "get_llm", lambda: _CheckerLLM(scenario["checker"])),
        patch.object(wf_mod, "verify_task", spy_verify),
        patch.object(sy_mod, "get_llm", lambda: _SynthLLM()),
    ):
        yield


def run_scenario(scenario: dict) -> ScenarioResult:
    from graph.workflow import build_workflow

    attempts: list[tuple[str, int]] = []
    with _patched_pipeline(scenario, attempts):
        app = build_workflow()
        result = app.invoke(
            {
                "query": scenario["query"],
                "tasks": [],
                "findings": [],
                "run_log": [],
                "current_task_id": None,
                "report": None,
                "mock_mode": False,
            }
        )
    tasks = result["tasks"]
    done = [t for t in tasks if t.status == TaskStatus.DONE]
    failed = [t for t in tasks if t.status == TaskStatus.FAILED]

    verified = [c for t in tasks for c in t.claims if c.verified]
    with_source = [c for c in verified if c.source]

    # Group claim counts per task, in attempt order; stable = never grows.
    by_task: dict[str, list[int]] = {}
    for tid, n in attempts:
        by_task.setdefault(tid, []).append(n)
    stable = all(
        all(b <= a for a, b in zip(counts, counts[1:])) for counts in by_task.values()
    )

    trap_caught = None
    if scenario["kind"] == "trap":
        trap_caught = bool(tasks) and len(failed) == len(tasks)

    return ScenarioResult(
        scenario_id=scenario["id"],
        kind=scenario["kind"],
        tasks_total=len(tasks),
        tasks_done=len(done),
        tasks_failed=len(failed),
        trap_caught=trap_caught,
        verified_claims=len(verified),
        verified_with_source=len(with_source),
        retry_stable=stable,
        attempts=list(by_task.values()),
    )


def summarize(results: list[ScenarioResult]) -> EvalSummary:
    traps = [r for r in results if r.kind == "trap"]
    trap_rate = (
        sum(1 for r in traps if r.trap_caught) / len(traps) if traps else None
    )
    v = sum(r.verified_claims for r in results)
    vs = sum(r.verified_with_source for r in results)
    precision = vs / v if v else None
    stability = (
        sum(1 for r in results if r.retry_stable) / len(results) if results else 1.0
    )
    return EvalSummary(
        trap_catch_rate=trap_rate,
        verification_precision=precision,
        retry_stability_rate=stability,
        scenarios=results,
    )


def run_scenarios(
    scenarios: list[dict], results_dir: str | None = None
) -> EvalSummary:
    """Run scenarios and optionally log the JSON summary to results_dir."""
    results = [run_scenario(s) for s in scenarios]
    summary = summarize(results)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = os.path.join(results_dir, f"eval-{stamp}.json")
        with open(path, "w") as f:
            json.dump(asdict(summary), f, indent=2)
        print(f"results -> {path}")
    return summary


def main() -> None:
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    summary = run_scenarios(SCENARIOS, results_dir=results_dir)
    print(f"{'scenario':<22}{'kind':<8}{'done':<6}{'failed':<8}{'trap?':<7}{'stable'}")
    for r in summary.scenarios:
        print(
            f"{r.scenario_id:<22}{r.kind:<8}{r.tasks_done:<6}{r.tasks_failed:<8}"
            f"{str(r.trap_caught):<7}{r.retry_stable}"
        )
    print(f"\ntrap_catch_rate:        {summary.trap_catch_rate}")
    print(f"verification_precision: {summary.verification_precision}")
    print(f"retry_stability_rate:   {summary.retry_stability_rate}")


if __name__ == "__main__":
    main()
