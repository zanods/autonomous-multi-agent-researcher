from __future__ import annotations

import os


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Tavily web search when TAVILY_API_KEY is set, else [].

    Each hit: {"title": ..., "snippet": ..., "url": ...}.
    Free tier: 1,000 searches/month at tavily.com.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    res = client.search(query, max_results=max_results, search_depth="advanced")
    return [
        {
            "title": r.get("title", ""),
            "snippet": (r.get("content", "") or "")[:500],
            "url": r.get("url", ""),
        }
        for r in res.get("results", [])
    ]
