"""RAG helper functions for loading a PDF, creating a vector store and
retrieving relevant context.

This module implements the standard Retrieval‑Augmented Generation workflow:

1. Split the document into manageable chunks.
2. Generate embeddings and store them in a vector database (Qdrant).
3. Perform similarity search to retrieve the most relevant chunks for a query.
4. Use the retrieved context when answering questions【190255345688846†L42-L56】.
"""

from __future__ import annotations

import os
from typing import List, Iterable, Optional

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
# from qdrant_client.http.models import VectorParams, Distance
from PyPDF2 import PdfReader


load_dotenv()


def load_pdf_documents(pdf_path: str) -> List[Document]:
    """Load a PDF file and return a list of LangChain ``Document`` objects.

    Each page in the PDF becomes a separate ``Document`` with the page index
    stored in the metadata.  If the PDF has no text, returns an empty list.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    List[Document]
        A list of documents containing page text and metadata.
    """
    reader = PdfReader(pdf_path)
    docs: List[Document] = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        docs.append(Document(page_content=text, metadata={"page": i}))
    return docs


def split_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Split a list of documents into smaller chunks for indexing.

    Uses a ``RecursiveCharacterTextSplitter`` to ensure that chunks respect
    linguistic boundaries and have some overlap to preserve context【190255345688846†L60-L100】.

    Parameters
    ----------
    documents : List[Document]
        Documents to split.
    chunk_size : int, default 1000
        Maximum characters per chunk.
    chunk_overlap : int, default 200
        Number of overlapping characters between chunks.

    Returns
    -------
    List[Document]
        A list of split documents.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def create_qdrant_vector_store(
    splits: List[Document],
    embeddings: OpenAIEmbeddings,
    *,
    collection_name: str = "pdf_collection",
    persist_path: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
):
    """
    Build a Qdrant vector store.
    - If qdrant_url is provided, connect to a server (Docker/Cloud).
    - Otherwise, use LOCAL on-disk mode via `path=...` (no server).
    """
    if qdrant_url:
        # Remote/server mode
        vs = QdrantVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            url=qdrant_url,
            api_key=qdrant_api_key or None,
            collection_name=collection_name,
        )
    else:
        if not persist_path:
            raise ValueError("persist_path is required for local Qdrant mode")
        # Ensure the directory exists (important for first run)
        os.makedirs(persist_path, exist_ok=True)

        # Local, on-disk mode (no server)
        vs = QdrantVectorStore.from_documents(
            documents=splits,
            embedding=embeddings,
            path=persist_path,               # <-- key change: use path= (not a client instance)
            collection_name=collection_name,
        )
    return vs


def retrieve_context(vector_store: QdrantVectorStore, query: str, k: int = 5) -> str:
    """Retrieve top‑k relevant chunks from Qdrant and join them into a single context string.

    Parameters
    ----------
    vector_store : QdrantVectorStore
        The vector store to query.
    query : str
        The user's question.
    k : int, default 5
        Number of similar documents to retrieve.

    Returns
    -------
    str
        Concatenated content of the retrieved documents.
    """
    docs = vector_store.similarity_search(query, k=k)
    contents = []
    for doc in docs:
        contents.append(doc.page_content)
    return "\n\n".join(contents)