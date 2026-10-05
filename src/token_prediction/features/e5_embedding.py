from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

E5_MODEL_ID = "intfloat/e5-small-v2"
E5_EMBEDDING_DIM = 384


def _default_workspace_root() -> Path:


    override = os.environ.get("TOKEN_PREDICTION_WORKSPACE_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "workspace"


E5_LOCAL_DIR = _default_workspace_root() / "models" / "e5-small-v2"
DEFAULT_CACHE_PATH = _default_workspace_root() / "cache" / "e5_small_v2.npz"

_MODEL = None
_TOKENIZER = None


E5_PREFIXES = ("passage", "query")


def _text_key(text: str, prefix: str = "passage") -> str:
    return hashlib.sha256(f"{prefix}: {text}".encode("utf-8")).hexdigest()


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        if not E5_LOCAL_DIR.is_dir():
            raise RuntimeError(f"no local e5-small-v2 snapshot at {E5_LOCAL_DIR}")
        from transformers import AutoModel, AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(E5_LOCAL_DIR, local_files_only=True)
        _MODEL = AutoModel.from_pretrained(E5_LOCAL_DIR, local_files_only=True).to("cpu").eval()
    return _MODEL, _TOKENIZER


def _encode_uncached(
    texts: Sequence[str], *, batch_size: int = 32, prefix: str = "passage"
) -> list[tuple[float, ...]]:


    import torch

    torch.set_num_threads(1)
    model, tokenizer = _load_model()
    if prefix not in E5_PREFIXES:
        raise ValueError(f"e5 prefix must be one of {E5_PREFIXES}")
    prefixed = [f"{prefix}: {text}" for text in texts]
    vectors: list[tuple[float, ...]] = []
    with torch.no_grad():
        for start in range(0, len(prefixed), batch_size):
            batch = prefixed[start : start + batch_size]
            encoded = tokenizer(
                batch, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            )
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(tuple(float(v) for v in row) for row in normalized)
    return vectors


class E5EmbeddingCache:


    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        self._vectors: dict[str, tuple[float, ...]] = {}
        if self.path.exists():
            import numpy as np

            with np.load(self.path) as archive:
                for key, vector in zip(archive["keys"], archive["vectors"]):
                    self._vectors[str(key)] = tuple(float(v) for v in vector)

    def embed(self, texts: Sequence[str], *, prefix: str = "passage") -> list[tuple[float, ...]]:


        keys = [_text_key(text, prefix) for text in texts]
        missing = list(dict.fromkeys(text for text, key in zip(texts, keys) if key not in self._vectors))
        if missing:
            for text, vector in zip(missing, _encode_uncached(missing, prefix=prefix)):
                self._vectors[_text_key(text, prefix)] = vector
            self._save()
        return [self._vectors[key] for key in keys]

    def _save(self) -> None:
        import numpy as np

        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._vectors)
        vectors = np.asarray([self._vectors[key] for key in keys], dtype=np.float32)
        np.savez(self.path, keys=np.asarray(keys, dtype=str), vectors=vectors)


_DEFAULT_CACHE: E5EmbeddingCache | None = None


def embed_texts(
    texts: Sequence[str], *, cache_path: Path | str | None = None, prefix: str = "passage"
) -> list[tuple[float, ...]]:


    global _DEFAULT_CACHE
    if cache_path is not None:
        return E5EmbeddingCache(cache_path).embed(texts, prefix=prefix)
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = E5EmbeddingCache()
    return _DEFAULT_CACHE.embed(texts, prefix=prefix)


__all__ = [
    "E5_EMBEDDING_DIM",
    "E5_MODEL_ID",
    "E5EmbeddingCache",
    "DEFAULT_CACHE_PATH",
    "embed_texts",
]
