"""Feature 5 — auto-language output: the report follows the query's language."""

from models.task import Claim, Task, TaskStatus
from optimization.language import detect_language


def test_detect_language():
    assert detect_language("Nhiệt độ CPU giảm bao nhiêu khi undervolt?") == "vi"
    assert detect_language("So sánh tản nhiệt buồng hơi và kim loại lỏng") == "vi"
    assert detect_language("What is the worst-case complexity of quicksort?") == "en"
    assert detect_language("") == "en"
    # Plain toned Latin (shared with French/Spanish) is not Vietnamese.
    assert detect_language("A café with naïve décor") == "en"


def test_synthesizer_live_prompt_directs_vietnamese(monkeypatch):
    from agents import synthesizer

    seen = {}

    class FakeLLM:
        def invoke(self, prompt):
            seen["prompt"] = prompt
            return type("Resp", (), {"content": "Báo cáo [1]."})()

    monkeypatch.setattr(synthesizer, "get_llm", lambda: FakeLLM())
    task = Task(
        title="Undervolt",
        status=TaskStatus.DONE,
        claims=[
            Claim(
                text="Undervolt giúp giảm nhiệt độ CPU.",
                source="https://v.example/u",
                verified=True,
            )
        ],
        source_titles={"https://v.example/u": "Undervolt Guide"},
    )
    synthesizer.synthesize(
        "Undervolt giúp giảm nhiệt độ CPU bao nhiêu?", [task], mock_mode=False
    )
    prompt = seen["prompt"]
    assert "tiếng Việt" in prompt
    assert "## Nguồn" in prompt  # localized Sources section directive
    assert "Missing evidence" in prompt  # contract marker unchanged


def test_synthesizer_english_unchanged():
    from agents import synthesizer

    failed = Task(title="Bad task", status=TaskStatus.FAILED)
    report = synthesizer.synthesize("q", [failed], mock_mode=True)
    assert "Verification gaps" in report
    assert "No research tasks could be verified" in report


def test_synthesizer_vietnamese_gaps_localized():
    from agents import synthesizer

    failed = Task(title="Task lỗi", status=TaskStatus.FAILED)
    report = synthesizer.synthesize(
        "Nhiệt độ CPU là bao nhiêu?", [failed], mock_mode=True
    )
    assert "Khoảng trống kiểm chứng" in report
    assert "Không có task nghiên cứu nào được kiểm chứng" in report
