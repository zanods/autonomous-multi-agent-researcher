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


_NEGATIONS = re.compile(r"\b(no|not|never|without|cannot)\b|n't")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _keyword_cover(claim: str, text: str) -> bool:
    """Pure keyword-subset check, ignoring negation — the raw signal used
    by contradiction detection."""
    kws = _keywords(claim)
    return bool(kws) and kws <= _words(text)


def _chunk_covers(claim: str, text: str) -> bool:
    """Auto-pass gate: keyword cover with no negation anywhere.

    Negation in the CLAIM ("does not reduce") or in the EVIDENCE chunk
    ("does not reduce") makes keyword matching ambiguous — "X reduces temps"
    would otherwise auto-pass against "X does not reduce temps". Those go
    to LLM review instead. Strict on purpose: anything string matching
    can't prove goes to stage 2.
    """
    if _NEGATIONS.search(claim.lower()) or _NEGATIONS.search(text.lower()):
        return False
    return _keyword_cover(claim, text)


def _match_chunk(claim: str, evidence: list[str]) -> int | None:
    """Index of the first evidence chunk covering the claim's keywords.

    Pure string matching — costs zero LLM tokens. Strict on purpose:
    anything it can't prove goes to stage 2 (LLM review).
    """
    for i, ev in enumerate(evidence):
        if _chunk_covers(claim, ev):
            return i
    return None


_CONTRAST = re.compile(
    r"\b(but|however|although|though|whereas|while|despite|in contrast|"
    r"on the other hand|contrary to)\b",
    re.IGNORECASE,
)


def _contradicted(claim: str, evidence: list[str]) -> bool:
    """True if the evidence disagrees with itself about the claim.

    Some chunk fully covers the claim's keywords (a would-be auto-pass)
    while another chunk covers at least half of them AND carries a negation
    or contrast marker — the claim is cherry-picked from disagreeing
    sources and must not pass on keyword matching alone. Routed to LLM
    review instead, which can adjudicate.
    """
    kws = _keywords(claim)
    if not kws:
        return False
    if not any(_keyword_cover(claim, t) for t in evidence):
        return False  # nothing fully covers it — nothing to cherry-pick
    for t in evidence:
        if _keyword_cover(claim, t):
            continue
        overlap = len(kws & _words(t)) / len(kws)
        if overlap >= 0.5 and (_NEGATIONS.search(t.lower()) or _CONTRAST.search(t.lower())):
            return True
    return False


def exact_match_verify(claim: str, evidence: list[str]) -> bool:
    """True if every keyword of the claim appears inside a single evidence chunk."""
    return _match_chunk(claim, evidence) is not None


def attach_claim_sources(
    claims: list[str], evidence: list[tuple[str | None, str, str]]
) -> list["Claim"]:
    """Attach the best-matching evidence ref to each claim.

    `evidence` is a list of (ref, title, text) triples — ref is a URL,
    "kb:<chunk_id>", or None when unknown. The first chunk whose keywords
    cover the claim's keywords wins; claims with no match (or a match on an
    unknown ref) get source=None and must never be cited. Local import of
    Claim avoids a models->optimization import cycle.
    """
    from models.task import Claim

    texts = [t for _, _, t in evidence]
    out = []
    for c in claims:
        idx = _match_chunk(c, texts)
        src = evidence[idx][0] if idx is not None else None
        out.append(Claim(text=c, source=src or None))
    return out
