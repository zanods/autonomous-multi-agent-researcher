from __future__ import annotations

import re

_STOPWORDS = set(
    "the a an and or of to in on for with is are was were be as at by from "
    "that this it its into than then so such no not can will would should "
    "could their there here which who whom whose what when where how why".split()
)


def split_claims(notes: str) -> list[str]:
    """Split research notes into candidate claims (sentences). Zero LLM tokens."""
    parts = re.split(r"(?<=[.!?])\s+", (notes or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _keywords(claim: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", claim.lower())
    return {w for w in words if len(w) > 3 and w not in _STOPWORDS}


def exact_match_verify(claim: str, evidence: list[str]) -> bool:
    """True if every keyword of the claim appears inside a single evidence chunk.

    Pure string matching — costs zero LLM tokens. Strict on purpose: anything
    it can't prove goes to stage 2 (LLM review).
    """
    kws = _keywords(claim)
    if not kws:
        return False
    for ev in evidence:
        ev_words = set(re.findall(r"[a-z0-9]+", ev.lower()))
        if kws <= ev_words:
            return True
    return False
