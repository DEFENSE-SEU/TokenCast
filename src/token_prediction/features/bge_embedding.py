from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
BGE_EMBEDDING_DIM = 384
DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[3] / "workspace" / "cache" / "bge_small_en_v1_5.npz"
)

_MODEL = None
_TOKENIZER = None


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _local_snapshot_dir() -> Path:


    from huggingface_hub.constants import HF_HUB_CACHE

    root = Path(HF_HUB_CACHE) / f"models--{BGE_MODEL_ID.replace('/', '--')}" / "snapshots"
    snapshots = sorted(p for p in root.iterdir() if p.is_dir())
    if not snapshots:
        raise RuntimeError(f"no cached snapshot for {BGE_MODEL_ID}; expected one under {root}")
    return snapshots[-1]


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        from transformers import AutoModel, AutoTokenizer

        snapshot = _local_snapshot_dir()
        _TOKENIZER = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        _MODEL = AutoModel.from_pretrained(snapshot, local_files_only=True).to("cpu").eval()
    return _MODEL, _TOKENIZER


def _encode_uncached(texts: Sequence[str], *, batch_size: int = 32) -> list[tuple[float, ...]]:


    import torch

    torch.set_num_threads(1)
    model, tokenizer = _load_model()
    vectors: list[tuple[float, ...]] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch, padding="max_length", truncation=True, max_length=512, return_tensors="pt"
            )
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.extend(tuple(float(v) for v in row) for row in normalized)
    return vectors


class BgeEmbeddingCache:


    def __init__(self, path: Path | str = DEFAULT_CACHE_PATH) -> None:
        self.path = Path(path)
        self._vectors: dict[str, tuple[float, ...]] = {}
        if self.path.exists():
            import numpy as np

            with np.load(self.path) as archive:
                for key, vector in zip(archive["keys"], archive["vectors"]):
                    self._vectors[str(key)] = tuple(float(v) for v in vector)

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:


        keys = [_text_key(text) for text in texts]
        missing = list(dict.fromkeys(text for text, key in zip(texts, keys) if key not in self._vectors))
        if missing:
            for text, vector in zip(missing, _encode_uncached(missing)):
                self._vectors[_text_key(text)] = vector
            self._save()
        return [self._vectors[key] for key in keys]

    def _save(self) -> None:
        import numpy as np

        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._vectors)
        vectors = np.asarray([self._vectors[key] for key in keys], dtype=np.float32)
        np.savez(self.path, keys=np.asarray(keys, dtype=str), vectors=vectors)


_DEFAULT_CACHE: BgeEmbeddingCache | None = None


def embed_texts(
    texts: Sequence[str], *, cache_path: Path | str | None = None
) -> list[tuple[float, ...]]:


    global _DEFAULT_CACHE
    if cache_path is not None:
        return BgeEmbeddingCache(cache_path).embed(texts)
    if _DEFAULT_CACHE is None:
        _DEFAULT_CACHE = BgeEmbeddingCache()
    return _DEFAULT_CACHE.embed(texts)


__all__ = [
    "BGE_EMBEDDING_DIM",
    "BGE_MODEL_ID",
    "BgeEmbeddingCache",
    "DEFAULT_CACHE_PATH",
    "embed_texts",
]
