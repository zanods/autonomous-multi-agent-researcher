"""Adversarial verifier cases: negation, unit changes, out-of-context quotes,
contradictory sources, and bogus citations must all FAIL."""

from eval.run_eval import run_scenarios
from eval.trap_queries import SCENARIOS
from models.task import Claim, Task, TaskStatus
from optimization.claims import _contradicted, _match_chunk


def _adversarial():
    return [s for s in SCENARIOS if s["kind"] == "adversarial"]


def test_all_adversarial_scenarios_caught():
    """The 5 adversarial trap categories all FAIL in the mocked eval."""
    summary = run_scenarios(_adversarial(), results_dir=None)
    assert {r.scenario_id for r in summary.scenarios} == {
        "trap-negation",
        "trap-units",
        "trap-halfquote",
        "trap-contradiction",
        "trap-bogus-citation",
    }
    assert summary.trap_catch_rate == 1.0
    for r in summary.scenarios:
        assert r.trap_caught is True, r.scenario_id
        assert r.tasks_failed == r.tasks_total > 0


def test_negated_claim_never_exact_matches():
    ev = ["Vapor chambers spread heat laterally across heatsinks."]
    assert (
        _match_chunk("Vapor chambers do not spread heat laterally.", ev) is None
    )


def test_negated_evidence_never_exact_matches():
    # The dangerous direction: positive claim vs negated evidence must not
    # auto-pass ("X reduces temps" vs "X does not reduce temps").
    ev = ["Independent tests show vapor chambers do not reduce temperatures."]
    assert _match_chunk("Vapor chambers reduce temperatures.", ev) is None


def test_unit_change_never_exact_matches():
    ev = ["Graphene pads achieve a temperature drop of 5 degrees Celsius."]
    assert (
        _match_chunk(
            "Graphene pads achieve a temperature drop of 5 degrees Fahrenheit.",
            ev,
        )
        is None
    )


def test_halfquote_never_exact_matches():
    ev = ["The study found vapor chambers ineffective without proper airflow."]
    assert _match_chunk("The study found vapor chambers effective.", ev) is None


_CON_A = (
    "Vapor chambers reduce temperatures by 15 degrees Celsius "
    "according to vendor tests."
)
_CON_B = (
    "Independent tests show vapor chambers do not reduce temperatures "
    "significantly."
)
_CON_CLAIM = "Vapor chambers reduce temperatures by 15 degrees Celsius."


def test_contradiction_detected():
    assert _contradicted(_CON_CLAIM, [_CON_A, _CON_B]) is True


def test_no_contradiction_single_source():
    assert _contradicted(_CON_CLAIM, [_CON_A]) is False


def test_no_contradiction_without_coverage():
    assert (
        _contradicted("Quantum batteries charge instantly.", [_CON_A, _CON_B])
        is False
    )


def test_contradictory_claim_goes_to_llm_not_autopass(monkeypatch):
    """A cherry-picked claim must reach the LLM stage, not auto-pass.

    Without contradiction routing, chunk A would exact-match and the LLM
    would never be consulted.
    """
    import agents.fact_checker as fc_mod

    ref_a = "https://www.intel.com/vapor-chamber-whitepaper"
    ref_b = "https://www.tomshardware.com/vapor-chamber-test"
    task = Task(title="Contradiction task", status=TaskStatus.RESEARCHED)
    task.claims = [Claim(text=_CON_CLAIM)]
    task.evidence = [ref_a, ref_b]
    task.evidence_texts = [_CON_A, _CON_B]
    task.evidence_credibility = {ref_a: 0.80, ref_b: 0.65}

    seen = {}

    class _FailLLM:
        def invoke(self, prompt):
            seen["prompt"] = prompt
            return type("Resp", (), {"content": "1. FAIL - sources disagree."})()

    monkeypatch.setattr(fc_mod, "get_llm", lambda: _FailLLM())
    ok, _note, failed = fc_mod.verify_task(task, mock_mode=False)

    assert "prompt" in seen, "LLM stage was skipped — claim auto-passed"
    assert ok is False
    assert failed == [_CON_CLAIM]


def test_fact_checker_prompt_names_adversarial_failures(monkeypatch):
    """The LLM-verify prompt explicitly covers the adversarial categories."""
    import agents.fact_checker as fc_mod

    task = Task(title="Prompt check", status=TaskStatus.RESEARCHED)
    task.claims = [Claim(text="Some unverifiable claim about cooling.")]
    task.evidence_texts = ["Unrelated evidence about battery chemistry."]

    seen = {}

    class _CaptureLLM:
        def invoke(self, prompt):
            seen["prompt"] = prompt
            return type("Resp", (), {"content": "1. FAIL"})()

    monkeypatch.setattr(fc_mod, "get_llm", lambda: _CaptureLLM())
    fc_mod.verify_task(task, mock_mode=False)

    prompt = seen["prompt"]
    assert "opposite of the evidence" in prompt
    assert "changes units or numbers" in prompt
    assert "out of context" in prompt
