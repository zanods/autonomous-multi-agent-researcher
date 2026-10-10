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
