from typing import List, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np


class EmbeddingPipeline:
    """
    Handles document chunking and embedding generation for the RAG pipeline
    using a local SentenceTransformer model.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name

        print(f"[INFO] Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    # ---------------------------------------------------------------------
    # Split documents into manageable chunks
    # ---------------------------------------------------------------------
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    # ---------------------------------------------------------------------
    # Generate embeddings for document chunks
    # ---------------------------------------------------------------------
    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype=np.float32)

        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------
if __name__ == "__main__":
    from src.data_loader import load_all_documents

    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline(
        model_name="all-MiniLM-L6-v2",
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print(f"[INFO] Generated embeddings for {len(chunks)} chunks.")
