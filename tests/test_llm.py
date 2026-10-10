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
