# from dotenv import load_dotenv
# import os
# from langchain_groq import ChatGroq

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ENV_PATH = os.path.join(BASE_DIR, ".env")

# load_dotenv(dotenv_path=ENV_PATH)
# #print("[DEBUG] Key (inside app):", os.getenv("GROQ_API_KEY"))
# print("[DEBUG] Key loaded:", api_key)

# llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-20b")
# resp = llm.invoke("Explain what this model is in one sentence.")
# print(resp.content)

from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from src.vectorstore import FaissVectorStore  # your vectorstore module

# ------------------------------------------------------
# Load .env
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

# ------------------------------------------------------
# Initialize LLM
# ------------------------------------------------------
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-20b")

# ------------------------------------------------------
# Initialize Vector Store
# ------------------------------------------------------
vectorstore = FaissVectorStore(persist_dir="faiss_store", embedding_model="all-MiniLM-L6-v2")
vectorstore.load()  # assumes your FAISS index already exists

# ------------------------------------------------------
# Query Vector Store
# ------------------------------------------------------
query = "Explain what attention mechanism is. Tell me what you understood from data used for your training and what new you learned."
results = vectorstore.query(query, top_k=3)

# Extract text context from metadata
texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
context = "\n\n".join(texts)

if not context:
    print("No relevant context found.")
    exit()

# ------------------------------------------------------
# Build RAG Prompt
# ------------------------------------------------------
prompt = f"""
Use the following context to answer the question concisely.

Context:
{context}

Question: {query}

Answer:
"""

# ------------------------------------------------------
# Send to LLM
# ------------------------------------------------------
resp = llm.invoke(prompt)
print("\n[ANSWER]\n", resp.content)

