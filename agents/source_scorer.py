"""Heuristic credibility scoring for evidence sources.

Rule-based on purpose: transparent, dependency-free, and tunable without
retraining anything. Scores live in [0, 1]; higher means more trustworthy.

Bands:
  0.90  peer-reviewed / academic (journals, arxiv, .edu, ...)
  0.80  official vendor / developer docs
  0.70  user-curated local knowledge base (kb: chunks)
  0.65  reputable tech press
  0.55  wikipedia
  0.30  unknown blogs / domains
  0.15  unusable refs ("unknown", "web", unparseable)
"""

from __future__ import annotations

from urllib.parse import urlparse

_ACADEMIC_DOMAINS = frozenset(
    {
        "arxiv.org",
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "ieee.org",
        "acm.org",
        "nih.gov",
        "pubmed.ncbi.nlm.nih.gov",
        "doi.org",
        "plos.org",
        "springer.com",
        "wiley.com",
        "openreview.net",
    }
)
_ACADEMIC_SUFFIXES = (".edu", ".ac.uk", ".ac.jp", ".edu.au", ".ac.cn")

_VENDOR_DOMAINS = frozenset(
    {
        "intel.com",
        "amd.com",
        "nvidia.com",
        "microsoft.com",
        "kernel.org",
        "python.org",
        "developer.mozilla.org",
    }
)

_TECH_PRESS = frozenset(
    {
        "tomshardware.com",
        "anandtech.com",
        "techpowerup.com",
        "pcgamer.com",
        "phoronix.com",
    }
)

SCORE_ACADEMIC = 0.90
SCORE_VENDOR = 0.80
SCORE_LOCAL_KB = 0.70
SCORE_TECH_PRESS = 0.65
SCORE_WIKI = 0.55
SCORE_UNKNOWN = 0.30
SCORE_UNUSABLE = 0.15

# A claim whose supporting evidence all scores below this is "weakly
# sourced": it must not pass verification on that evidence alone.
LOW_CREDIBILITY = 0.40


def _domain(ref: str) -> str:
    try:
        host = urlparse(ref).netloc.lower()
    except Exception:
        return ""
    return host[4:] if host.startswith("www.") else host


def score_source(ref: str | None, title: str = "") -> float:
    """Heuristic credibility score in [0, 1] for an evidence reference.

    `ref` is a URL, "kb:<chunk_id>", or None/"unknown". `title` is accepted
    for future use (e.g. journal-name matching) and currently unused.
    """
    if not ref or ref in ("unknown", "web"):
        return SCORE_UNUSABLE
    if ref.startswith("kb:"):
        # User-curated local knowledge base: trusted, but not peer-reviewed.
        return SCORE_LOCAL_KB
    dom = _domain(ref)
    if not dom:
        return SCORE_UNUSABLE
    if dom in _ACADEMIC_DOMAINS or dom.endswith(_ACADEMIC_SUFFIXES):
        return SCORE_ACADEMIC
    if dom in _VENDOR_DOMAINS or dom.startswith("docs."):
        return SCORE_VENDOR
    if dom in _TECH_PRESS:
        return SCORE_TECH_PRESS
    if dom == "wikipedia.org" or dom.endswith(".wikipedia.org"):
        return SCORE_WIKI
    return SCORE_UNKNOWN
