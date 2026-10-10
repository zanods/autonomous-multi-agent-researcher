from agents import llm as llm_mod


def test_no_key_no_llm(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm_mod.get_llm() is None
    assert llm_mod.has_api_key() is False


def test_google_preferred_over_openai(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    assert llm_mod._active_provider() == "google"
    assert llm_mod.has_api_key() is True


def test_groq_fallback(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm_mod._active_provider() == "groq"
    assert llm_mod.has_api_key() is True


def test_openrouter_fallback(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm_mod._active_provider() == "openrouter"
    assert llm_mod.has_api_key() is True


class _ExplodingLLM:
    """Simulates a model whose structured-output path blows up."""

    def with_structured_output(self, schema):
        raise RuntimeError("tool calling not supported")

    def invoke(self, prompt):
        class R:
            content = '{"tasks": [{"title": "Task One", "description": "Desc One"}, {"title": "Task Two", "description": "Desc Two"}]}'

        return R()


def test_planner_json_fallback(monkeypatch):
    import agents.planner as planner_mod

    monkeypatch.setattr(planner_mod, "get_llm", lambda: _ExplodingLLM())
    tasks = planner_mod.plan_tasks("some query", mock_mode=False)
    assert len(tasks) == 2
    assert tasks[0].title == "Task One"
