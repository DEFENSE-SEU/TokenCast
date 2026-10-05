from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from functools import lru_cache


TEXT_EMBEDDING_DIM = 256
TEXT_EMBEDDING_VERSION_ID = "hashed-signed-logtf-v1"
PREFIX_TAIL_CHARS = 512
CHAR_NGRAM_ORDER = 3
PREFIX_COMPOSITION_FEATURE_NAMES = (
    "prefix_char_count", "prefix_word_count", "prefix_word_chars",
    "prefix_digit_count", "prefix_whitespace_count", "prefix_punctuation_count",
    "prefix_nonascii_count", "prefix_utf8_per_char", "prefix_mean_word_chars",
)

_WORD = re.compile(r"[\w]+", re.UNICODE)
_WHITESPACE_RUN = re.compile(r"\S+")


def prefix_composition_features(text: str) -> dict[str, float | int]:
    words = _WORD.findall(text)
    word_chars = sum(map(len, words))
    return {
        "prefix_char_count": len(text),
        "prefix_word_count": len(words),
        "prefix_word_chars": word_chars,
        "prefix_digit_count": sum(char.isdigit() for char in text),
        "prefix_whitespace_count": sum(char.isspace() for char in text),
        "prefix_punctuation_count": sum(not char.isalnum() and not char.isspace() for char in text),
        "prefix_nonascii_count": sum(not char.isascii() for char in text),
        "prefix_utf8_per_char": len(text.encode("utf-8")) / max(len(text), 1),
        "prefix_mean_word_chars": word_chars / max(len(words), 1),
    }


def _terms(text: str) -> tuple[str, ...]:
    words = tuple(token.lower() for token in _WORD.findall(text))
    bigrams = tuple(f"{left}::{right}" for left, right in zip(words, words[1:]))
    return words + bigrams


@lru_cache(maxsize=512)
def hashed_text_embedding(
    text: str,
    *,
    dim: int = TEXT_EMBEDDING_DIM,
) -> tuple[float, ...]:


    if dim <= 0:
        raise ValueError("embedding dimension must be positive")
    weights = [0.0] * dim
    for term, count in Counter(_terms(text)).items():
        digest = zlib.crc32(term.encode("utf-8"))
        bucket = digest % dim
        sign = 1.0 if (digest >> 16) & 1 else -1.0
        weights[bucket] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(value * value for value in weights))
    if norm == 0.0:
        return tuple(weights)
    return tuple(value / norm for value in weights)


def _hashed_embedding(terms: Counter[str], dim: int) -> tuple[float, ...]:
    weights = [0.0] * dim
    for term, count in terms.items():
        digest = zlib.crc32(term.encode("utf-8"))
        bucket = digest % dim
        sign = 1.0 if (digest >> 16) & 1 else -1.0
        weights[bucket] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(value * value for value in weights))
    if norm == 0.0:
        return tuple(weights)
    return tuple(value / norm for value in weights)


@lru_cache(maxsize=512)
def hashed_char_ngram_embedding(
    text: str,
    *,
    order: int = CHAR_NGRAM_ORDER,
    dim: int = TEXT_EMBEDDING_DIM,
) -> tuple[float, ...]:


    if dim <= 0 or order <= 0:
        raise ValueError("embedding dimension and n-gram order must be positive")
    folded = " ".join(text.lower().split())
    grams = Counter(folded[index : index + order] for index in range(max(0, len(folded) - order + 1)))
    return _hashed_embedding(grams, dim)


def prefix_tail_embeddings(visible_text_prefix: str) -> dict[str, tuple[float, ...]]:


    tail = visible_text_prefix[-PREFIX_TAIL_CHARS:]
    return {
        "generation_prefix_tail_embedding": hashed_text_embedding(tail),
        "generation_prefix_tail_char_embedding": hashed_char_ngram_embedding(tail),
    }


def task_text_shape_features(task_text: str) -> dict[str, object]:


    return {
        "task_char_count": len(task_text),
        "task_word_count": len(_WHITESPACE_RUN.findall(task_text)),
        "task_line_count": max(1, len(task_text.splitlines())),
        "task_code_fence_count": task_text.count("```") // 2,
        "task_embedding": hashed_text_embedding(task_text),
    }
