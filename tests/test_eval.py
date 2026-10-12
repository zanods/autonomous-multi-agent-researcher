"""Feature 3 — eval harness: a tiny mocked eval must compute all three metrics."""

from eval.run_eval import run_scenarios
from eval.trap_queries import SCENARIOS


def _tiny():
    return [s for s in SCENARIOS if s["id"] in ("trap-fractalhash", "normal-quicksort")]


def test_tiny_eval_computes_metrics():
    summary = run_scenarios(_tiny(), results_dir=None)
    assert summary.trap_catch_rate == 1.0
    assert summary.verification_precision == 1.0
    assert summary.retry_stability_rate == 1.0


def test_trap_scenario_fails_and_normal_verifies():
    summary = run_scenarios(_tiny(), results_dir=None)
    by_id = {r.scenario_id: r for r in summary.scenarios}
    trap = by_id["trap-fractalhash"]
    assert trap.trap_caught is True
    assert trap.tasks_failed == 1
    assert trap.verified_claims == 0
    normal = by_id["normal-quicksort"]
    assert normal.tasks_done == 1
    assert normal.verified_claims == normal.verified_with_source == 1
