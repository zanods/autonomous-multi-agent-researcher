from agents import llm as llm_mod


def test_no_key_no_llm(monkeypatch):
    for key in (
        "GOOGLE_API_KEY",
        "GROQ_API_KEY",
        "CEREBRAS_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)
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


def test_openai_compatible_providers_set_request_timeout(monkeypatch):
    """Free-tier congestion must fail fast instead of hanging on the ~10min SDK default."""
    # Sanitize proxy env: the workspace NO_PROXY has bracketed IPv6 entries
    # that break httpx client construction.
    for var in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    for env_key in ("GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv(env_key, "fake")
        llm = llm_mod.get_llm()
        assert llm.request_timeout == 90, env_key


def test_cerebras_selection_and_timeout(monkeypatch):
    for var in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("CEREBRAS_API_KEY", "fake")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm_mod._active_provider() == "cerebras"
    assert llm_mod.has_api_key() is True
    llm = llm_mod.get_llm()
    assert llm.request_timeout == 90


def test_message_text_handles_block_content():
    """Google-style providers return content as a list of blocks, not a string."""

    class _BlockMsg:
        content = [
            {"type": "text", "text": "Vapor chamber reduces hotspot temps "},
            {"type": "text", "text": "by 5-10C."},
        ]

    class _StrMsg:
        content = "plain string"

    assert llm_mod.message_text(_BlockMsg()) == "Vapor chamber reduces hotspot temps by 5-10C."
    assert llm_mod.message_text(_StrMsg()) == "plain string"
    assert llm_mod.message_text("raw") == "raw"
