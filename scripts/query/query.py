import chromadb
import sys
import os

# Add the project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embed.embedding_model import model, collection  # Now it should work

# Connect to ChromaDB (running in Docker)
chroma_client = chromadb.HttpClient(host="localhost", port=9000)
collection = chroma_client.get_collection(name="legal_docs")

def query_chroma(user_query, top_k=5):
    """Search ChromaDB for similar legal cases."""
    query_embedding = model.encode([user_query])  # Convert query to embedding

    # Perform similarity search
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k  # Get top N relevant cases
    )

    # Extract retrieved documents & metadata
    if results["documents"]:
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            print(f"\n🔹 **Match {i+1}**")
            print(f"📜 Case Number: {meta.get('caseNumber', 'Unknown')}")
            print(f"📖 Document: {doc[:500]}...")  # Show first 500 chars
    else:
        print("\n❌ No relevant legal cases found.")

# Example query
if __name__ == "__main__":
    user_query = input("\n🔎 Enter your legal question: ")
    query_chroma(user_query, top_k=3)
