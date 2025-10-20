"""Streamlit chat interface for the AI pipeline.

This script starts a Streamlit app that loads the RAG vector store, builds
the LangGraph pipeline and handles user interactions.  Chat history is kept
in ``st.session_state`` and displayed using ``st.chat_message()``, with a
simple input box provided by ``st.chat_input()``【875021349489622†L1476-L1485】.

Run this file with ``streamlit run src/ui_app.py`` after installing the
dependencies listed in ``requirements.txt``.
"""

from __future__ import annotations

import os
import pathlib
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from src.rag import load_pdf_documents, split_documents, create_qdrant_vector_store
from src.graph import build_pipeline


# Load environment variables
load_dotenv()

print("DEBUG OPENWEATHER_API_KEY:", os.getenv("OPENWEATHER_API_KEY"))

@st.cache_resource
def init_resources() -> tuple:
    """Initialise embeddings, vector store and the LangGraph pipeline once.

    Returns a tuple of (pipeline, embeddings).  Streamlit caches the result
    so the expensive initialisation is only run once per session.
    """
    # Location of the sample PDF.  You can replace this with another file.
    data_dir = pathlib.Path(__file__).resolve().parent.parent / "data"
    pdf_path = data_dir / "sample.pdf"
    if not pdf_path.exists():
        # Attempt to generate a simple sample PDF so the app can run without
        # external files.  Try to use the `fpdf` library if available; fall back
        # to creating a blank page.  This is only executed on first run.
        try:
            from fpdf import FPDF  # type: ignore

            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            sample_text = (
                "This is a sample PDF document used for demonstrating RAG.\n\n"
                "The Eiffel Tower is located in Paris, France. It is one of the most famous landmarks in the world.\n\n"
                "Weather information is often needed by travellers. The weather in Hyderabad can be very hot in the summer."
            )
            pdf.multi_cell(0, 10, sample_text)
            pdf.output(str(pdf_path))
        except Exception:
            # Final fallback: create a blank PDF
            from PyPDF2 import PdfWriter

            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            with open(pdf_path, "wb") as f:
                writer.write(f)
    # Build embeddings and vector store
    embeddings = OpenAIEmbeddings()
    documents = load_pdf_documents(str(pdf_path))
    splits = split_documents(documents)
    # Determine whether to connect to a remote Qdrant instance or use local
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url:
        vector_store = create_qdrant_vector_store(
            splits,
            embeddings,
            collection_name="pdf_collection",
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )
    else:
        persist_path = str(data_dir / "qdrant")
        vector_store = create_qdrant_vector_store(
            splits,
            embeddings,
            collection_name="pdf_collection",
            persist_path=persist_path,
        )
    # Create language model
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    llm = ChatOpenAI(model=model_name, temperature=0)
    # Build pipeline
    pipeline = build_pipeline(llm, vector_store)
    return pipeline, embeddings


def main() -> None:
    st.title("Weather & PDF Assistant")
    st.markdown("Ask me about the weather in a city or anything contained in the PDF.")
    # Initialise resources
    pipeline, _ = init_resources()
    # Initialise chat history in session state
    if "history" not in st.session_state:
        st.session_state.history = []
    # Display previous messages
    for role, content in st.session_state.history:
        st.chat_message(role).write(content)
    # Chat input
    prompt = st.chat_input("Enter your question")
    if prompt:
        # Add user message to history and display it
        st.session_state.history.append(("human", prompt))
        st.chat_message("human").write(prompt)
        # Invoke pipeline
        result = pipeline.invoke({"question": prompt})
        answer = result.get("answer") or result.get("context") or ""
        st.session_state.history.append(("ai", answer))
        st.chat_message("ai").write(answer)


if __name__ == "__main__":
    main()