# embedding_model.py

import chromadb
from transformers import AutoTokenizer, AutoModel
import torch

# 1) Connect to Chroma (running in Docker on localhost:9000)
chroma_client = chromadb.HttpClient(host="localhost", port=9000)

# 2) Create or get a collection
collection_name = "legal_docs"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=None
)

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained("upskyy/bge-m3-korean")
hf_model = AutoModel.from_pretrained("upskyy/bge-m3-korean")
hf_model.eval()

device = "cuda"
hf_model.to(device)

# Define embedding function (mean pooling)
def embed_texts(texts, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        encoded_input = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)

        with torch.no_grad():
            model_output = hf_model(**encoded_input)

        embeddings = model_output.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()