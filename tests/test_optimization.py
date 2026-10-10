from agents.fact_checker import verify_task
from models.task import Task
from optimization.budget import TokenBudget
from optimization.claims import exact_match_verify, split_claims
from optimization.tokens import count_tokens, truncate_to_tokens


def test_count_and_truncate():
    assert count_tokens("hello world") > 0
    long = "word " * 5000
    assert count_tokens(long) > 1000
    cut = truncate_to_tokens(long, 100)
    assert count_tokens(cut) <= 150  # heuristic tolerance


def test_budget_spend_trims_and_tracks():
    b = TokenBudget(limits={"researcher": 50})
    original = "word " * 5000
    out = b.spend("researcher", original)
    assert count_tokens(out) < count_tokens(original)
    assert b.usage["researcher"] > 0
    assert b.total_tokens() == b.usage["researcher"]


def test_budget_no_trim_under_limit():
    b = TokenBudget(limits={"researcher": 100000})
    text = "short text"
    assert b.spend("researcher", text) == text


def test_exact_match_verify():
    ev = ["Vapor chambers spread heat laterally across the heatsink in gaming laptops."]
    assert exact_match_verify("Vapor chambers spread heat across the heatsink.", ev)
    assert not exact_match_verify("Quantum batteries charge in five seconds.", ev)
    assert not exact_match_verify("Hi.", ev)  # too short -> no keywords


def test_split_claims():
    claims = split_claims("First claim here is definitely long enough. Second claim also long enough!")
    assert len(claims) == 2
    assert split_claims("") == []


def test_verify_mock_unchanged():
    t = Task(title="Sample task", result="Some notes here that are long enough.")
    ok, note = verify_task(t, mock_mode=True)
    assert ok and note == "mock-verified"


def test_cost_estimate_non_negative():
    b = TokenBudget()
    b.spend("researcher", "hello world")
    assert b.estimated_cost_usd("gpt-4o-mini") >= 0
    assert "total" in b.report()


def test_exact_match_rejects_negated_claims():
    """A negated claim must never auto-pass stage 1, even when every other
    keyword overlaps the evidence."""
    ev = ["Liquid metal thermal compound is recommended for copper heatsinks."]
    # All non-negation keywords overlap, but the claim is negated.
    assert not exact_match_verify(
        "Liquid metal thermal compound is not recommended for copper heatsinks.", ev
    )
    assert not exact_match_verify(
        "Liquid metal thermal compound isn't recommended for copper heatsinks.", ev
    )
    # The affirmative version still passes.
    assert exact_match_verify(
        "Liquid metal thermal compound is recommended for copper heatsinks.", ev
    )


def test_staged_verify_rejects_empty_llm_reply(monkeypatch):
    """Vacuous pass: if the LLM returns nothing, unverified claims must fail."""
    import agents.fact_checker as fc_mod

    task = Task(
        title="Sample task",
        result="Vapor chambers spread heat laterally across gaming laptop heatsinks.",
    )
    task.evidence_texts = ["Completely unrelated evidence about battery chemistry."]

    class _EmptyLLM:
        def invoke(self, prompt):
            class R:
                content = ""
            return R()

    monkeypatch.setattr(fc_mod, "get_llm", lambda: _EmptyLLM())
    ok, note = fc_mod.verify_task(task, mock_mode=False)
    assert ok is False
    assert "0/" in note


def test_measured_cost_splits_input_output():
    b = TokenBudget()

    class _Msg:
        usage_metadata = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}

    b.record("researcher", _Msg())
    # gpt-4o-mini: $0.15/1M in + $0.60/1M out = $0.75
    assert b.measured_cost_usd("gpt-4o-mini") == 0.75
    assert b.measured_cost_usd("unknown-model") == 0.0
