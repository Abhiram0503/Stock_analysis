# import os
# from dotenv import load_dotenv
# from src.vectorstore import FaissVectorStore
# from langchain_groq import ChatGroq
# print("[DEBUG] search.py loaded successfully")


# load_dotenv()

# class RAGSearch:
#     def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "openai/gpt-oss-20b"):
#         self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
#         # Load or build vectorstore
#         faiss_path = os.path.join(persist_dir, "faiss.index")
#         meta_path = os.path.join(persist_dir, "metadata.pkl")
#         if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
#             from data_loader import load_all_documents
#             docs = load_all_documents("data")
#             self.vectorstore.build_from_documents(docs)
#         else:
#             self.vectorstore.load()
#         print(f"[DEBUG] Loaded GROQ_API_KEY: {groq_api_key}")
#         self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
#         print(f"[INFO] Groq LLM initialized: {llm_model}")

#     def search_and_summarize(self, query: str, top_k: int = 5) -> str:
#         results = self.vectorstore.query(query, top_k=top_k)
#         texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
#         context = "\n\n".join(texts)
#         if not context:
#             return "No relevant documents found."
#         prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
#         response = self.llm.invoke([prompt])
#         return response.content

# # Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "What is attention mechanism?"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)


import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

print("[DEBUG] search.py loaded successfully")

# -------------------------------------------------------------------
# Load environment variables from project root (absolute path)
# -------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

print("[DEBUG] Loaded env from:", ENV_PATH)

class RAGSearch:
    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "openai/gpt-oss-20b"
    ):
        # -------------------------------
        # Initialize vectorstore
        # -------------------------------
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

        # -------------------------------
        # Initialize Groq LLM
        # -------------------------------
        groq_api_key = os.getenv("GROQ_API_KEY")
        print(f"[DEBUG] Loaded GROQ_API_KEY: {groq_api_key}")

        if not groq_api_key:
            raise ValueError("❌ GROQ_API_KEY not found in environment variables")

        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] ✅ Groq LLM initialized: {llm_model}")

    # -------------------------------------------------------------------
    # Search and summarize
    # -------------------------------------------------------------------
    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""Summarize the following context for the query: '{query}'.

        Context:
        {context}

        Summary:"""

        # ✅ FIXED: invoke with a string, not a list
        print("[DEBUG] Sending prompt to Groq LLM...")
        response = self.llm.invoke(prompt)

        # ✅ Handle both object and string outputs safely
        if hasattr(response, "content"):
            return response.content
        else:
            return str(response)


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("\nSummary:\n", summary)
