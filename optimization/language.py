"""Lightweight query-language detection (no heavy deps).

Vietnamese is detected by characters unique to it: đ/Đ and the breve /
circumflex / horn vowel families (ă â ê ô ơ ư plus tone marks). Plain toned
Latin (é, ñ, ...) is shared with French/Spanish, so it does NOT trigger —
e.g. "café" still counts as English. Any real Vietnamese sentence contains
several core characters, so a single occurrence is enough signal.
"""

from __future__ import annotations

_VI_CORE = set(
    "đĐ"
    "ăằắặẳẵĂẰẮẶẲẴ"
    "âầấậẩẫÂẦẤẬẨẪ"
    "êềếệểễÊỀẾỆỂỄ"
    "ôồốộổỗÔỒỐỘỔỖ"
    "ơờớợởỡƠỜỚỢỞỠ"
    "ưừứựửữƯỪỨỰỬỮ"
)


def detect_language(text: str) -> str:
    """Return 'vi' for Vietnamese, 'en' otherwise (default)."""
    if not text:
        return "en"
    return "vi" if any(c in _VI_CORE for c in text) else "en"
