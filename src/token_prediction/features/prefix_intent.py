from __future__ import annotations

import re

PREFIX_INTENT_VERSION_ID = "prefix-intent-seed-v1"
PREFIX_INTENT_TAIL_CHARS = 240

CONTINUATION_MARKERS: tuple[str, ...] = (
    "first",
    "next",
    "then",
    "let me",
    "let's",
    "now",
    "also",
    "additionally",
    "step",
    "here is",
    "here's",
    "we need to",
    "i'll",
    "i will",
    "to do this",
)
WRAPUP_MARKERS: tuple[str, ...] = (
    "in summary",
    "to summarize",
    "in conclusion",
    "finally",
    "done",
    "this completes",
    "the fix",
    "hope this helps",
    "let me know",
    "overall",
    "that's it",
)

PREFIX_INTENT_FEATURE_NAMES: tuple[str, ...] = (
    "prefix_open_code_fence",
    "prefix_code_fence_count",
    "prefix_open_list_item",
    "prefix_last_line_chars",
    "prefix_ends_sentence",
    "prefix_ends_colon",
    "prefix_line_count",
    "prefix_tail_continuation_markers",
    "prefix_tail_wrapup_markers",
)

_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
_WORD_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}


def _marker_pattern(marker: str) -> re.Pattern[str]:
    pattern = _WORD_BOUNDARY_CACHE.get(marker)
    if pattern is None:
        pattern = re.compile(r"(?<![a-z'])" + re.escape(marker) + r"(?![a-z'])")
        _WORD_BOUNDARY_CACHE[marker] = pattern
    return pattern


def _marker_hits(tail: str, markers: tuple[str, ...]) -> int:
    return sum(len(_marker_pattern(marker).findall(tail)) for marker in markers)


def generation_prefix_intent_features(text: str) -> dict[str, int]:


    fences = text.count("```")
    lines = text.split("\n")
    last_line = lines[-1]
    last_nonempty = next((line for line in reversed(lines) if line.strip()), "")
    stripped = text.rstrip()
    tail = text[-PREFIX_INTENT_TAIL_CHARS:].lower()
    return {
        "prefix_open_code_fence": fences % 2,
        "prefix_code_fence_count": fences // 2,
        "prefix_open_list_item": int(bool(_LIST_ITEM.match(last_nonempty))),
        "prefix_last_line_chars": len(last_line),
        "prefix_ends_sentence": int(stripped.endswith((".", "!", "?"))),
        "prefix_ends_colon": int(stripped.endswith(":")),
        "prefix_line_count": len(lines),
        "prefix_tail_continuation_markers": _marker_hits(tail, CONTINUATION_MARKERS),
        "prefix_tail_wrapup_markers": _marker_hits(tail, WRAPUP_MARKERS),
    }


__all__ = [
    "CONTINUATION_MARKERS",
    "PREFIX_INTENT_FEATURE_NAMES",
    "PREFIX_INTENT_TAIL_CHARS",
    "PREFIX_INTENT_VERSION_ID",
    "WRAPUP_MARKERS",
    "generation_prefix_intent_features",
]
