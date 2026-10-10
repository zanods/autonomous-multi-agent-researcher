from __future__ import annotations


def count_tokens(text: str) -> int:
    """Token count via tiktoken (cl100k) when available, else ~4 chars/token."""
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, len(text) // 4)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut text down to ~max_tokens, backing off to a word boundary."""
    if count_tokens(text) <= max_tokens:
        return text
    approx_chars = max_tokens * 4
    cut = text[:approx_chars]
    for sep in (". ", "\n", " "):
        idx = cut.rfind(sep)
        if idx > approx_chars * 0.5:
            cut = cut[: idx + len(sep)].rstrip()
            break
    return cut + "\n…[truncated]"
