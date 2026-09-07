"""Small deterministic model-output signals; no LLM judge."""

from __future__ import annotations

import collections
import unicodedata
from typing import Sequence


def repetition_rate(tokens: Sequence[int | str] | None) -> float | None:
    if not tokens:
        return None
    count = max(collections.Counter(tokens).values())
    return count / len(tokens)


def punctuation_spam_rate(text: str | None) -> float | None:
    if text is None:
        return None
    visible = [char for char in text if not char.isspace()]
    if not visible:
        return 0.0
    punctuation = sum(unicodedata.category(char).startswith("P") for char in visible)
    return punctuation / len(visible)


def obvious_gibberish(text: str | None) -> bool | None:
    if text is None:
        return None
    visible = [char for char in text if char.isalnum()]
    if not visible:
        return False
    scripts: set[str] = set()
    for char in visible:
        name = unicodedata.name(char, "")
        for script in ("LATIN", "CJK", "HANGUL", "ARABIC", "CYRILLIC"):
            if script in name:
                scripts.add(script)
                break
    return len(visible) >= 8 and len(scripts) >= 3


def first_difference(left: Sequence[object] | None, right: Sequence[object] | None) -> int | None:
    if left is None or right is None:
        return None
    for index, (left_item, right_item) in enumerate(zip(left, right)):
        if left_item != right_item:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None
