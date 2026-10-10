"""Tests for TokenBudget.record() — API-reported usage tracking."""

from optimization.budget import TokenBudget


class _Msg:
    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}


def test_record_usage_metadata_style():
    b = TokenBudget()
    b.record("researcher", _Msg(usage_metadata={"input_tokens": 100, "output_tokens": 25, "total_tokens": 125}))
    b.record("researcher", _Msg(usage_metadata={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60}))
    assert b.measured["researcher"] == {"input": 150, "output": 35}
    assert b.total_measured() == 185


def test_record_openai_token_usage_style():
    b = TokenBudget()
    msg = _Msg(response_metadata={"token_usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100}})
    b.record("fact_checker", msg)
    assert b.measured["fact_checker"] == {"input": 80, "output": 20}


def test_record_no_metadata_is_noop():
    b = TokenBudget()
    b.record("planner", _Msg())
    b.record("planner", object())
    assert b.measured == {}
    assert b.total_measured() == 0


def test_report_prefers_measured():
    b = TokenBudget()
    b.spend("researcher", "hello world")  # heuristic estimate
    b.record("researcher", _Msg(usage_metadata={"input_tokens": 1000, "output_tokens": 200}))
    report = b.report()
    assert "[measured]" in report
    assert "1200 tokens" in report
    assert "(in 1000 / out 200)" in report


def test_report_falls_back_to_estimate():
    b = TokenBudget()
    b.spend("researcher", "hello world")
    report = b.report()
    assert "[estimated]" in report
    assert "[measured]" not in report


def test_planner_records_structured_output_usage(monkeypatch):
    import agents.planner as planner_mod
    from agents.planner import TaskPlan, TaskDraft

    plan = TaskPlan(tasks=[TaskDraft(title="Task One", description="Desc One")])
    msg = _Msg(usage_metadata={"input_tokens": 300, "output_tokens": 40})

    class _Structured:
        def invoke(self, prompt):
            return {"raw": msg, "parsed": plan, "parsing_error": None}

    class _LLM:
        def with_structured_output(self, schema, include_raw=False):
            assert include_raw is True
            return _Structured()

    monkeypatch.setattr(planner_mod, "get_llm", lambda: _LLM())
    budget = TokenBudget()
    tasks = planner_mod.plan_tasks("some query", mock_mode=False, budget=budget)
    assert len(tasks) == 1
    assert budget.measured["planner"] == {"input": 300, "output": 40}
