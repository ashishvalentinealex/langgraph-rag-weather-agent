"""Tests for the LangGraph pipeline."""

import tempfile, time, numpy as np
from typing import Any
from langchain_core.documents import Document

from src.graph import build_pipeline, GraphState
from src.rag import create_qdrant_vector_store


# -----------------------------
# Fake Embeddings (no API call)
# -----------------------------
class FakeEmbeddings:
    """Simple deterministic embeddings with fixed vector size for tests."""
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


# -----------------------------
# Fake Model (simulates LLM)
# -----------------------------
class FakeModel:
    """Fake ChatOpenAI replacement for testing decision routing."""
    def invoke(self, messages: Any) -> Any:
        # messages[-1] is a HumanMessage usually
        text = getattr(messages[-1], "content", str(messages[-1]))
        if "weather" in text.lower():
            class Resp: content = "weather"
            return Resp()
        else:
            class Resp: content = "Paris is the capital of France."
            return Resp()


# -----------------------------
# The actual test
# -----------------------------
def test_pipeline_decision():
    """Ensure pipeline routes correctly between weather and PDF paths."""
    docs = [Document(page_content="Paris is the capital of France.", metadata={})]
    embeddings = FakeEmbeddings()

    tmp_path = tempfile.mkdtemp(prefix=f"qdrant_test_{int(time.time())}_")
    vec_store = create_qdrant_vector_store(
        docs, embeddings, persist_path=tmp_path, collection_name="decision_test"
    )

    pipeline = build_pipeline(FakeModel(), vec_store)

    # Weather question
    result_weather: GraphState = pipeline.invoke({"question": "What is the weather in Paris?"})
    assert result_weather.get("route") == "weather"
    assert result_weather.get("answer") is not None

    # Non-weather (PDF) question
    result_pdf: GraphState = pipeline.invoke({"question": "What is the capital of France?"})
    assert result_pdf.get("route") == "pdf"
    assert "Paris" in (result_pdf.get("answer") or "")
