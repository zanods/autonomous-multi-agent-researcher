"""Feature 1 — inline citations: claims carry source refs from researcher
through fact-checker into the synthesized report's numbered Sources section."""

from models.task import Claim, Task, TaskStatus
from optimization.claims import attach_claim_sources


def test_attach_claim_sources_picks_covering_chunk():
    evidence = [
        ("https://a.example/cpu", "CPU guide", "Completely unrelated text about pasta recipes."),
        ("kb:chunk-7", "chunk-7", "Vapor chambers spread heat laterally across gaming laptop heatsinks."),
    ]
    claims = attach_claim_sources(
        ["Vapor chambers spread heat laterally across gaming laptop heatsinks."],
        evidence,
    )
    assert len(claims) == 1
    assert claims[0].source == "kb:chunk-7"


def test_attach_claim_sources_none_when_no_match():
    evidence = [("https://a.example/x", "X", "Text about pasta recipes and nothing else.")]
    claims = attach_claim_sources(["Quantum tunneling stabilizes fusion reactors."], evidence)
    assert claims[0].source is None


def test_fact_checker_records_verifying_source(monkeypatch):
    import agents.fact_checker as fc_mod

    task = Task(
        title="Sample task",
        result="Vapor chambers spread heat laterally across gaming laptop heatsinks.",
        evidence_texts=[
            "Unrelated pasta text.",
            "Vapor chambers spread heat laterally across gaming laptop heatsinks.",
        ],
        evidence=["https://pasta.example", "https://cooling.example/vc"],
    )
    # No LLM needed: exact match should verify and record the matched ref.
    class _NoLLM:
        def invoke(self, prompt):
            raise AssertionError("LLM must not be called on the exact-match path")

    monkeypatch.setattr(fc_mod, "get_llm", lambda: _NoLLM())
    ok, note, failed = fc_mod.verify_task(task, mock_mode=False, optimize=True)
    assert ok is True
    assert failed == []
    assert len(task.claims) == 1
    assert task.claims[0].verified is True
    assert task.claims[0].source == "https://cooling.example/vc"


def test_cited_claim_survives_to_report_with_number():
    from agents import synthesizer

    task = Task(
        title="Cooling",
        status=TaskStatus.DONE,
        claims=[
            Claim(text="Vapor chambers spread heat well.", source="https://c.example/vc", verified=True),
            Claim(text="An unsourced observation.", source=None, verified=True),
            Claim(text="A failed claim.", source="https://c.example/vc", verified=False),
        ],
        source_titles={"https://c.example/vc": "Cooling Guide"},
    )
    report = synthesizer.synthesize("q", [task], mock_mode=True)
    # Sourced claim gets citation [1] ...
    assert "Vapor chambers spread heat well. [1]" in report
    # ... sourceless claim gets no citation number ...
    assert "- An unsourced observation.\n" in report
    assert "An unsourced observation. [" not in report
    # ... failed claim is excluded entirely ...
    assert "A failed claim." not in report
    # ... and the Sources section maps [1] to title + URL.
    assert "## Sources" in report
    assert "[1] Cooling Guide — https://c.example/vc" in report


def test_citation_numbers_dedupe_across_tasks():
    from agents import synthesizer

    shared = "https://shared.example/doc"
    t1 = Task(
        title="Task One", status=TaskStatus.DONE,
        claims=[Claim(text="First claim here.", source=shared, verified=True)],
        source_titles={shared: "Shared Doc"},
    )
    t2 = Task(
        title="Task Two", status=TaskStatus.DONE,
        claims=[Claim(text="Second claim here.", source=shared, verified=True)],
        source_titles={shared: "Shared Doc"},
    )
    numbers, _titles = synthesizer.build_citation_map([t1, t2])
    assert numbers == {shared: 1}
    report = synthesizer.synthesize("q", [t1, t2], mock_mode=True)
    assert report.count("[1] Shared Doc — https://shared.example/doc") == 1


def test_synthesizer_live_prompt_has_citation_rules(monkeypatch):
    from agents import synthesizer

    seen = {}

    class FakeLLM:
        def invoke(self, prompt):
            seen["prompt"] = prompt
            return type("Resp", (), {"content": "Report [1]."})()

    monkeypatch.setattr(synthesizer, "get_llm", lambda: FakeLLM())
    task = Task(
        title="Good task",
        status=TaskStatus.DONE,
        claims=[Claim(text="Verified stuff here.", source="https://g.example", verified=True)],
        source_titles={"https://g.example": "Good Source"},
    )
    synthesizer.synthesize("q", [task], mock_mode=False)
    prompt = seen["prompt"]
    assert "citation" in prompt.lower()
    assert "[1]" in prompt
    assert "Missing evidence" in prompt  # verified-findings-only contract intact
    assert "do not cover these topics" in prompt.lower()
