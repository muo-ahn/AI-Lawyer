import chromadb
import subprocess
import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from embed.embedding_model import embed_texts
from dotenv import load_dotenv

load_dotenv()

os.environ["PATH"] += os.pathsep + os.getenv("OLLAMA_PATH")

# Connect to ChromaDB
chroma_client = chromadb.HttpClient(host="localhost", port=9000)
collection = chroma_client.get_collection(name="legal_docs")

def load_prompt_template():
    """Load the Ollama prompt template from a file."""
    prompt_path = os.path.join(os.path.dirname(__file__), "ollama_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()

def query_chroma(user_query, top_k=5):
    """Search ChromaDB for similar legal cases and return structured answer."""
    query_embedding = embed_texts([user_query])  # Convert query to embedding

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

        # Load the prompt template
        prompt_template = load_prompt_template()

        # Format the prompt dynamically
        ollama_prompt = prompt_template.format(
            user_query=user_query,
            retrieved_cases="\n".join(retrieved_texts)
        )

        print("\n📢 **DEBUG: Final Ollama Prompt**")
        print(ollama_prompt)

        # Call Ollama
        response = subprocess.run(
            ["ollama", "run", "llama2", "--system", "당신의 응답은 반드시 한국어여야 합니다.", ollama_prompt],
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
