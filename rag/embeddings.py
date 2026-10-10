from __future__ import annotations

import hashlib
import math
import os

from langchain_core.embeddings import Embeddings


class MockEmbeddings(Embeddings):
    """Deterministic hash-based embeddings for offline dev/test. No API needed.

    Bag-of-words hashed into a fixed dim vector, L2-normalized. Good enough
    to exercise the RAG plumbing; swap for OpenAIEmbeddings via get_embeddings().
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def get_embeddings() -> Embeddings:
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        )
    return MockEmbeddings()
