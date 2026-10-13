from __future__ import annotations

import os
import threading

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


# ChromaDB's PersistentClient is not safe to construct concurrently against
# the same directory: racing threads hit "Could not connect to tenant
# default_tenant" during tenant validation. The parallel-tasks feature calls
# load_vectorstore from several worker threads at once, so serialize client
# creation behind a lock and reuse one client per directory+embedding setup.
_client_lock = threading.Lock()
_client_cache: dict[tuple[str, str], Chroma] = {}


def load_vectorstore(persist_dir: str) -> Chroma:
    with _client_lock:
        embeddings = get_embeddings()
        key = (os.path.abspath(persist_dir), type(embeddings).__name__)
        vs = _client_cache.get(key)
        if vs is None:
            vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            _client_cache[key] = vs
        return vs
