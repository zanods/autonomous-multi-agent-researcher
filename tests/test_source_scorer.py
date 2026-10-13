"""Feature 2 — source credibility scoring: heuristic bands and the
fact-checker weighting rule (weakly-sourced claims must not verify)."""

from agents.source_scorer import LOW_CREDIBILITY, score_source
from models.task import Task


def test_scorer_bands():
    assert score_source("https://arxiv.org/abs/1234.5678") == 0.90
    assert score_source("https://www.nature.com/articles/x") == 0.90
    assert score_source("https://cs.stanford.edu/paper") == 0.90
    assert score_source("https://www.intel.com/content/www/us/en/docs") == 0.80
    assert score_source("kb:chunk-42") == 0.70
    assert score_source("https://www.tomshardware.com/reviews/x") == 0.65
    assert score_source("https://en.wikipedia.org/wiki/Cache") == 0.55
    assert score_source("https://randomblog.xyz/post") == 0.30
    assert score_source("unknown") == 0.15
    assert score_source(None) == 0.15
    assert score_source("not a url at all") == 0.15


def test_low_credibility_threshold_sane():
    assert LOW_CREDIBILITY == 0.40


def _task_with_scored_evidence(score: float) -> Task:
    return Task(
        title="Sample task",
        result="Vapor chambers spread heat laterally across gaming laptop heatsinks.",
        evidence_texts=[
            "Vapor chambers spread heat laterally across gaming laptop heatsinks."
        ],
        evidence=["https://shadyblog.example/vc"],
        evidence_credibility={"https://shadyblog.example/vc": score},
    )


def test_weakly_sourced_claim_cannot_verify(monkeypatch):
    """A claim supported only by an UNUSABLE ref ("unknown", unparseable)
    fails even if the LLM would pass it — and the LLM is never consulted."""
    import agents.fact_checker as fc_mod

    class _EagerLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM must not be consulted for unusable refs")

    monkeypatch.setattr(fc_mod, "get_llm", lambda: _EagerLLM())
    task = _task_with_scored_evidence(0.15)
    ok, note, failed = fc_mod.verify_task(task, mock_mode=False, optimize=True)
    assert ok is False
    assert "weakly-sourced" in note
    assert len(failed) == 1


def test_unknown_domain_source_gets_llm_review(monkeypatch):
    """Regression (2026-10-13 compsci run): a claim supported by a real URL
    on an unknown domain (0.30 — e.g. Baeldung, GeeksforGeeks) must NOT
    auto-fail. The credibility allowlist cannot cover every field; the LLM
    judges whether the evidence supports the claim."""
    import agents.fact_checker as fc_mod

    class _PassLLM:
        def invoke(self, prompt):
            class R:
                content = "1. PASS"
            return R()

    monkeypatch.setattr(fc_mod, "get_llm", lambda: _PassLLM())
    task = _task_with_scored_evidence(0.30)
    ok, note, failed = fc_mod.verify_task(task, mock_mode=False, optimize=True)
    assert ok is True, note
    assert failed == []
    assert "0 weakly-sourced" in note


def test_high_credibility_source_verifies_normally(monkeypatch):
    import agents.fact_checker as fc_mod

    class _NoLLM:
        def invoke(self, prompt):
            raise AssertionError("exact-match path needs no LLM")

    monkeypatch.setattr(fc_mod, "get_llm", lambda: _NoLLM())
    task = _task_with_scored_evidence(0.90)
    ok, _note, failed = fc_mod.verify_task(task, mock_mode=False, optimize=True)
    assert ok is True
    assert failed == []


def test_no_credibility_info_keeps_legacy_behavior(monkeypatch):
    """Tasks without researcher-attached scores are not penalized."""
    import agents.fact_checker as fc_mod

    class _NoLLM:
        def invoke(self, prompt):
            raise AssertionError("exact-match path needs no LLM")

    monkeypatch.setattr(fc_mod, "get_llm", lambda: _NoLLM())
    task = Task(
        title="Sample task",
        result="Vapor chambers spread heat laterally across gaming laptop heatsinks.",
        evidence_texts=[
            "Vapor chambers spread heat laterally across gaming laptop heatsinks."
        ],
        evidence=["https://whatever.example/x"],
        # no evidence_credibility -> no penalty
    )
    ok, _note, failed = fc_mod.verify_task(task, mock_mode=False, optimize=True)
    assert ok is True
    assert failed == []


def test_researcher_attaches_credibility_scores(monkeypatch):
    import agents.researcher as rs_mod

    class _SpyLLM:
        def invoke(self, prompt):
            class R:
                content = "Some research notes about vapor chambers cooling performance."
            return R()

    monkeypatch.setattr(rs_mod, "get_llm", lambda: _SpyLLM())
    monkeypatch.setattr(
        rs_mod, "rag_search", lambda q, k=4: [{"id": "doc-1", "text": "kb text"}]
    )
    monkeypatch.setattr(
        rs_mod,
        "web_search",
        lambda q, max_results=5: [
            {
                "title": "Arxiv paper",
                "snippet": "arxiv snippet",
                "url": "https://arxiv.org/abs/9999.9999",
            }
        ],
    )
    task = Task(title="Sample task")
    rs_mod.research_task(task, mock_mode=False, optimize=False)
    assert task.evidence_credibility["kb:doc-1"] == 0.70
    assert task.evidence_credibility["https://arxiv.org/abs/9999.9999"] == 0.90
