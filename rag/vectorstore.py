from __future__ import annotations

import os

from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.embeddings import get_embeddings


def default_persist_dir() -> str:
    return os.getenv("CHROMA_DIR", "./chroma_db")


def build_vectorstore(
    texts: list[str],
    metadatas: list[dict] | None = None,
    persist_dir: str | None = None,
) -> Chroma:
    """Build a Chroma vector store. In-memory when persist_dir is None."""
    embeddings = get_embeddings()
    docs = [
        Document(
            page_content=t,
            metadata=metadatas[i] if metadatas else {},
        )
        for i, t in enumerate(texts)
    ]
    if persist_dir:
        return Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    return Chroma.from_documents(docs, embeddings)


def load_vectorstore(persist_dir: str) -> Chroma:
    return Chroma(persist_directory=persist_dir, embedding_function=get_embeddings())
