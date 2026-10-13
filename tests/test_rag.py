from rag.embeddings import MockEmbeddings
from rag.vectorstore import build_vectorstore


def test_mock_embeddings_deterministic():
    e = MockEmbeddings()
    assert e.embed_query("hello world") == e.embed_query("hello world")
    assert len(e.embed_query("x")) == 384
    assert e.embed_query("aaa") != e.embed_query("zzz")


def test_ingest_and_retrieve_roundtrip():
    vs = build_vectorstore(
        [
            "laptop cooling uses vapor chambers for high TDP processors",
            "bananas are a yellow fruit",
        ]
    )
    docs = vs.similarity_search("vapor chamber cooling tdp", k=1)
    assert len(docs) == 1
    assert "vapor" in docs[0].page_content


def test_rag_search_empty_when_nothing_ingested(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "empty_db"))
    from rag.retriever import rag_search

    assert rag_search("anything") == []


def test_rag_search_thread_safe_on_disk_store(tmp_path, monkeypatch):
    """Regression: concurrent rag_search used to crash ChromaDB tenant
    validation ("Could not connect to tenant default_tenant") because every
    thread built its own PersistentClient. Parallel tasks rely on this."""
    import concurrent.futures

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    db_dir = str(tmp_path / "concurrent_db")
    monkeypatch.setenv("CHROMA_DIR", db_dir)
    build_vectorstore(
        ["laptop cooling uses vapor chambers for high TDP processors"],
        persist_dir=db_dir,
    )

    from rag.retriever import rag_search
    from rag import vectorstore

    vectorstore._client_cache.clear()

    def work(i):
        return rag_search("vapor chamber cooling", k=1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        results = list(ex.map(work, range(8)))

    assert all(len(r) == 1 and "vapor" in r[0]["text"] for r in results)
