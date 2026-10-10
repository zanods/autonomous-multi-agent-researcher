from __future__ import annotations

import os


def _active_provider() -> str | None:
    # Google first when it works (generous free tier), then Groq (free, no card),
    # then OpenRouter (free :free models), then OpenAI (paid).
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return None


def has_api_key() -> bool:
    return _active_provider() is not None


def get_llm():
    """Chat model for agents, or None when no API key is configured.

    Agents fall back to deterministic mock outputs when this returns None,
    so the graph can be exercised end-to-end without spending API credits.
    """
    provider = _active_provider()
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("MODEL", "gemini-2.0-flash"), temperature=0.2
        )
    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("MODEL", "llama-3.3-70b-versatile"), temperature=0.2
        )
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        # Dedicated env var: the generic MODEL default (gemini-2.0-flash) is a
        # paid slug on OpenRouter, so OpenRouter gets its own default.
        # cohere/north-mini-code:free is the verified-working default as of
        # 2026-10-10 (gemma-4-31b-it:free was rate-limited on the shared pool).
        return ChatOpenAI(
            model=os.getenv("OPENROUTER_MODEL", "cohere/north-mini-code:free"),
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/zanods/autonomous-multi-agent-researcher",
                "X-Title": "autonomous-multi-agent-researcher",
            },
            temperature=0.2,
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("MODEL", "gpt-4o-mini"), temperature=0.2)
    return None
