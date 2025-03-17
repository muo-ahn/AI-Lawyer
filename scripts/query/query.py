# query/query.py

import chromadb
import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
import subprocess
from embed.embedding_model import model

# Connect to ChromaDB
chroma_client = chromadb.HttpClient(host="localhost", port=9000)
collection = chroma_client.get_collection(name="legal_docs")

def query_chroma(user_query, top_k=5):
    """Search ChromaDB for similar legal cases and return structured answer."""
    query_embedding = model.encode([user_query])  # Convert query to embedding

    # Perform similarity search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k  # Get top N relevant cases
    )

    if results["documents"]:
        retrieved_texts = []
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            case_number = meta.get("caseNumber", "Unknown")
            retrieved_texts.append(f"📜 Case {i+1}: {case_number}\n\"{doc[:500]}...\"")

        # Format the query for Ollama
        ollama_prompt = f"""SYSTEM: You are a legal assistant. Given legal precedents, provide an explanation.

{chr(10).join(retrieved_texts)}

USER QUESTION: "{user_query}"

ANSWER:
"""
        # Call Ollama
        response = subprocess.run(
            ["ollama", "run", "llama2", ollama_prompt],
            capture_output=True,
            text=True,
            encoding="utf-8"
        )

        print("\n📝 **Generated Legal Answer:**")
        print(response.stdout.strip())  # Print LLM-generated answer
    else:
        print("\n❌ No relevant legal cases found.")

# Example query
if __name__ == "__main__":
    user_query = input("\n🔎 Enter your legal question: ")
    query_chroma(user_query, top_k=3)
