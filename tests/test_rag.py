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
