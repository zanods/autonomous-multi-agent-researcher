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
    ok, note, _failed = verify_task(t, mock_mode=True)
    assert ok and note == "mock-verified"


def test_verify_no_llm_fails_closed(monkeypatch):
    """Regression: mock_mode=False with no LLM configured must NOT verify.

    Fail-open (returning True) would mark every claim 'verified' without
    any verification on a live run without API keys.
    """
    import agents.fact_checker as fc_mod

    t = Task(title="Sample task", result="Some notes here that are long enough.")
    monkeypatch.setattr(fc_mod, "get_llm", lambda: None)
    ok, note, _failed = fc_mod.verify_task(t, mock_mode=False)
    assert ok is False
    assert "no LLM configured" in note


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
    ok, note, _failed = fc_mod.verify_task(task, mock_mode=False)
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


def test_staged_verify_fail_not_flipped_by_pass_in_text(monkeypatch):
    """A FAIL verdict whose explanation contains the word PASS must stay FAIL."""
    import agents.fact_checker as fc_mod

    task = Task(
        title="Sample task",
        result="Vapor chambers spread heat laterally across gaming laptop heatsinks.",
    )
    task.evidence_texts = ["Completely unrelated evidence about battery chemistry."]

    class _TrickyLLM:
        def invoke(self, prompt):
            class R:
                content = "1: FAIL - the evidence does not PASS the bar for this claim."
            return R()

    monkeypatch.setattr(fc_mod, "get_llm", lambda: _TrickyLLM())
    ok, note, _failed = fc_mod.verify_task(task, mock_mode=False)
    assert ok is False
    assert "1 failed" in note


def test_staged_verify_accepts_both_verdict_formats(monkeypatch):
    """'1. PASS' and '1: PASS' are both valid verdict lines."""
    import agents.fact_checker as fc_mod

    task = Task(
        title="Sample task",
        result="Vapor chambers spread heat laterally across gaming laptop heatsinks.",
    )
    task.evidence_texts = ["Vapor chambers spread heat laterally across gaming laptop heatsinks."]

    class _PassLLM:
        def invoke(self, prompt):
            class R:
                content = "1. PASS"
            return R()

    # Force the claim past exact-match so the LLM path is exercised.
    monkeypatch.setattr(fc_mod, "exact_match_verify", lambda c, t: False)
    monkeypatch.setattr(fc_mod, "get_llm", lambda: _PassLLM())
    ok, note, _failed = fc_mod.verify_task(task, mock_mode=False)
    assert ok is True


def test_verify_task_returns_failed_claims(monkeypatch):
    """verify_task reports exactly which claims failed, for retry focus."""
    import agents.fact_checker as fc_mod

    task = Task(
        title="Sample task",
        result="Vapor chambers spread heat laterally across gaming laptop heatsinks. "
        "Quantum batteries charge in five seconds flat.",
    )
    task.evidence_texts = ["Vapor chambers spread heat laterally across gaming laptop heatsinks."]

    class _MixedLLM:
        def invoke(self, prompt):
            class R:
                # Only one claim needs LLM review (the other exact-matches);
                # judge it FAIL.
                content = "1: FAIL - no supporting evidence found anywhere."
            return R()

    monkeypatch.setattr(fc_mod, "get_llm", lambda: _MixedLLM())
    ok, note, failed = fc_mod.verify_task(task, mock_mode=False)
    assert ok is False
    assert len(failed) == 1
    assert "Quantum batteries" in failed[0]


def test_research_retry_narrows_to_failed_claims(monkeypatch):
    """On retry, the researcher prompt targets failed claims and keeps notes short."""
    import agents.researcher as rs_mod

    captured = {}

    class _SpyLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            class R:
                content = "Targeted notes."
            return R()

    monkeypatch.setattr(rs_mod, "get_llm", lambda: _SpyLLM())
    monkeypatch.setattr(rs_mod, "rag_search", lambda q, k=4: [])
    monkeypatch.setattr(rs_mod, "web_search", lambda q, max_results=5: [])

    task = Task(title="Sample task", description="desc")
    task.evidence_texts = ["old evidence"]
    rs_mod.research_task(
        task, mock_mode=False, focus_claims=["Claim about undervolting voltage range."]
    )
    assert "UNVERIFIED" in captured["prompt"]
    assert "undervolting voltage range" in captured["prompt"]
    assert "do not broaden the topic" in captured["prompt"]
    # Old evidence is kept, not discarded.
    assert task.evidence_texts == ["old evidence"]


def test_workflow_retry_stores_unverified_claims(monkeypatch):
    """fact_checker_node stashes failed claims so researcher_node can focus."""
    from graph.workflow import fact_checker_node
    from models.task import TaskStatus
    import agents.fact_checker as fc_mod

    task = Task(title="Sample task", result="Some long enough research notes here.")
    task.transition(TaskStatus.IN_PROGRESS)
    task.transition(TaskStatus.RESEARCHED)
    state = {
        "tasks": [task],
        "current_task_id": task.id,
        "mock_mode": True,
        "token_budget": None,
    }
    # Mock mode verifies everything: no unverified claims stored.
    fact_checker_node(state)
    assert task.status.value == "done"
    assert task.unverified_claims == []

    # Now force a failure with known failed claims.
    task2 = Task(title="Sample task 2", result="Some long enough research notes here.")
    task2.transition(TaskStatus.IN_PROGRESS)
    task2.transition(TaskStatus.RESEARCHED)
    state2 = {
        "tasks": [task2],
        "current_task_id": task2.id,
        "mock_mode": False,
        "token_budget": None,
    }
    monkeypatch.setattr(
        fc_mod, "verify_task", lambda *a, **k: (False, "staged: 0/1", ["the bad claim"])
    )
    import graph.workflow as wf_mod

    monkeypatch.setattr(wf_mod, "verify_task", fc_mod.verify_task)
    fact_checker_node(state2)
    assert task2.status.value == "needs_retry"
    assert task2.unverified_claims == ["the bad claim"]
