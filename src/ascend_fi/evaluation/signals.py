"""Small deterministic model-output signals; no LLM judge."""

from __future__ import annotations

import collections
import re
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


def token_ngram_repetition(tokens: Sequence[int | str] | None, *, n: int = 2) -> float | None:
    """Return the fraction of repeated fixed-size token n-grams."""

    if not tokens or len(tokens) < n:
        return None
    grams = [tuple(str(token) for token in tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams)


def periodic_loop(
    text_or_tokens: str | Sequence[int | str] | None,
    *,
    max_period: int = 8,
    min_repeats: int = 3,
) -> bool | None:
    """Detect a contiguous repeated token pattern without an LLM judge."""

    if text_or_tokens is None:
        return None
    tokens = (
        re.findall(r"\w+|[^\w\s]", text_or_tokens, flags=re.UNICODE)
        if isinstance(text_or_tokens, str)
        else [str(token) for token in text_or_tokens]
    )
    for start in range(len(tokens)):
        for period in range(1, min(max_period, len(tokens) - start) + 1):
            unit = tokens[start : start + period]
            repeats = 1
            while tokens[start + repeats * period : start + (repeats + 1) * period] == unit:
                repeats += 1
            if repeats >= min_repeats:
                return True
    return False


def longest_repeated_run(text_or_tokens: str | Sequence[int | str] | None) -> int | None:
    """Return the longest run of one identical lexical token."""

    if text_or_tokens is None:
        return None
    tokens = (
        re.findall(r"\w+|[^\w\s]", text_or_tokens, flags=re.UNICODE)
        if isinstance(text_or_tokens, str)
        else [str(token) for token in text_or_tokens]
    )
    if not tokens:
        return 0
    longest = current = 1
    for previous, token in zip(tokens, tokens[1:]):
        current = current + 1 if token == previous else 1
        longest = max(longest, current)
    return longest


def longest_punctuation_run(text: str | None) -> int | None:
    """Return the longest contiguous visible punctuation run."""

    if text is None:
        return None
    longest = current = 0
    for char in text:
        if unicodedata.category(char).startswith("P"):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest
