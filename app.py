from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore



def main():
    print("\n=== 📘 Starting RAG Pipeline ===\n")

    # Step 1: Load all PDFs/texts
    docs = load_all_documents("data")
    if not docs:
        print("[WARN] No documents found in 'data' folder.")
        return

    # Step 2: Chunk and embed documents
    emb_pipe = EmbeddingPipeline(model_name="all-MiniLM-L6-v2")
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)

    # Step 3: Build FAISS vector store
    store = FaissVectorStore()
    store.build_from_documents(chunks)

    # Step 4: Run a test query
    print("\n=== 🔍 Testing Query ===\n")
    query_text = "What is multiheaded attention? Give two answers, one from the data you have been tained on and the other from the pdf provided."
    results = store.query(query_text)

    for i, result in enumerate(results, start=1):
        print(f"\n[Result {i}]")
        print(result["metadata"]["text"][:500])
        print("-" * 80)

    print("\n✅ [DONE] RAG Pipeline completed successfully.")


if __name__ == "__main__":
    main()
