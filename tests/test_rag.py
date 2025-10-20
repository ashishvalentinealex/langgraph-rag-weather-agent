"""Tests for the RAG helper functions."""

import tempfile, time, numpy as np
from langchain_core.documents import Document
from src.rag import split_documents, create_qdrant_vector_store, retrieve_context


class FakeEmbeddings:
    """Simple deterministic embeddings with fixed vector size (no API calls)."""
    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed_documents(self, texts):
        return [self._make_vec(t) for t in texts]

    def embed_query(self, text):
        return self._make_vec(text)

    def _make_vec(self, text: str):
        arr = np.zeros(self.dim, dtype=float)
        for i, ch in enumerate(text.encode("utf-8")):
            arr[i % self.dim] += ch / 255.0
        return arr.tolist()


def test_split_and_retrieve(tmp_path):
    """Create a tiny local vector store and test retrieval works."""
    docs = [
        Document(page_content="The Eiffel Tower is in Paris.", metadata={"page": 0}),
        Document(page_content="The Colosseum is in Rome.", metadata={"page": 1}),
    ]

    splits = split_documents(docs, chunk_size=50, chunk_overlap=0)
    embeddings = FakeEmbeddings()

    tmp_qdrant = tempfile.mkdtemp(prefix=f"qdrant_test_{int(time.time())}_")

    vec_store = create_qdrant_vector_store(
        splits,
        embeddings,
        collection_name="test_collection",
        persist_path=tmp_qdrant,
    )

    context = retrieve_context(vec_store, "Where is the Eiffel Tower?", k=2)
    assert "Eiffel Tower" in context
    assert "Paris" in context
