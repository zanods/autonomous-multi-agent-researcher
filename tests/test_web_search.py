from tools.web_search import web_search


def test_web_search_no_key_returns_empty(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert web_search("anything") == []
