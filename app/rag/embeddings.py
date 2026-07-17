"""Embedding backends behind one interface.

Default is `onnx`: all-MiniLM-L6-v2 (384-dim) executed on ONNX Runtime, which
chromadb already depends on. It is the same model the sentence-transformers
build would give you, but without torch — the torch wheel drags in ~2.5 GB of
CUDA libraries (cuBLAS, cuDNN, NCCL, Triton) that are pure dead weight on the
CPU boxes this actually runs on.

`sentence_transformers` remains available for GPU hosts, but is not installed by
default; see requirements.txt. `hash` is a deterministic bag-of-words fallback
used in CI so tests never download a model.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import get_settings

_TOKEN = re.compile(r"[a-z0-9']+")


def _l2_normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class HashEmbedder(Embedder):
    """Hashing-trick embedder. Deterministic, offline, ~useful for exact overlap."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for token in _TOKEN.findall(text.lower()):
                idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
                vec[idx] += 1.0
            vectors.append(_l2_normalise(vec))
        return vectors


class OnnxMiniLMEmbedder(Embedder):
    """all-MiniLM-L6-v2 on ONNX Runtime, shipped inside chromadb.

    Downloads ~80 MB on first use and caches under ~/.cache/chroma.
    """

    dim = 384

    def __init__(self):
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

        self._fn = ONNXMiniLM_L6_V2()

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Chroma normalises already; doing it here keeps the cosine assumption in
        # vectorstore.py true regardless of what the backend hands back.
        return [_l2_normalise([float(x) for x in v]) for v in self._fn(texts)]


class SentenceTransformerEmbedder(Embedder):
    """Optional GPU path. Requires `pip install sentence-transformers`."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "EMBEDDING_BACKEND=sentence_transformers requires an extra install: "
                "`pip install sentence-transformers`. On CPU-only hosts prefer "
                "EMBEDDING_BACKEND=onnx, which uses the same model without torch."
            ) from exc

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).tolist()


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    if settings.embedding_backend == "hash":
        return HashEmbedder()
    if settings.embedding_backend == "sentence_transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    return OnnxMiniLMEmbedder()
