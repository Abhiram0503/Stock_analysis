"""
RAG (Retrieval-Augmented Generation) pipeline package.

This package provides:
- Document loading (PDF, TXT, CSV, Excel, DOCX, JSON)
- Text chunking and embedding generation
- FAISS vector storage and similarity search
- RAG-style search and summarization using Gemini (default)

Modules:
    data_loader.py  -> Load documents from multiple formats.
    embedding.py    -> Create and manage embeddings.
    vectorstore.py  -> Manage FAISS index and metadata.
    search.py       -> Perform retrieval-augmented LLM search.
"""

from .data_loader import load_all_documents
from .embedding import EmbeddingPipeline
from .vectorstore import FaissVectorStore
from .search import RAGSearch

__all__ = [
    "load_all_documents",
    "EmbeddingPipeline",
    "FaissVectorStore",
    "RAGSearch",
]
