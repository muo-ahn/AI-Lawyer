import chromadb

# Connect to Chroma
client = chromadb.HttpClient(host="localhost", port=9000)

# List all collections
collections = client.list_collections()
print("Available collections:")

# Delete the one you want (e.g., legal_docs)
client.delete_collection("legal_docs")
print("✅ Collection 'legal_docs' deleted.")
