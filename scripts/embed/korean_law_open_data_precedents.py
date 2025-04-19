# embed/korean_law_open_data_precedents.py

import uuid
import logging
from embedding_model import embed_texts, collection
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    filename="korean_law_open_data_precedents.log",
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Script started: korean_law_open_data_precedents.py")

try:
    dataset = load_dataset("joonhok-exo-ai/korean_law_open_data_precedents", split="train")
    logging.info("Successfully loaded Hugging Face dataset: korean_law_open_data_precedents")
except Exception as e:
    logging.error(f"Failed to load dataset: {e}")
    raise

# Define fields of interest
FIELDS_OF_INTEREST = [
    "판례정보일련번호", "사건명", "사건번호", "선고일자", "선고", "법원명",
    "사건종류명", "판결유형", "판시사항", "판결요지", "참조조문",
    "참조판례", "전문"
]

logging.info(f"Fields of interest: {FIELDS_OF_INTEREST}")

# Split long text into smaller chunks
def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Process and insert one batch of the dataset
def process_and_insert(dataset_slice, embed_batch_size=64):
    text_chunks = []
    metadata_list = []

    for entry in dataset_slice:
        combined_text = []
        meta = {}

        for field in FIELDS_OF_INTEREST:
            if field in entry and entry[field]:
                field_text = str(entry[field]).strip()
                combined_text.append(f"{field}: {field_text}")
                if field == "판례정보일련번호":
                    meta["case_id"] = field_text

        if combined_text:
            text_chunk = "\n".join(combined_text)
            text_chunks.append(text_chunk)
            metadata_list.append(meta)

    logging.info(f"Extracted {len(text_chunks)} text chunks.")

    # Chunk long text if needed
    final_chunks = []
    final_metadata = []

    for t, m in zip(text_chunks, metadata_list):
        if len(t) > 1000:
            for i, sub_t in enumerate(chunk_text(t)):
                new_meta = dict(m)
                new_meta["chunk_index"] = i
                final_chunks.append(sub_t)
                final_metadata.append(new_meta)
        else:
            final_chunks.append(t)
            final_metadata.append(m)

    logging.info(f"Total chunks after splitting: {len(final_chunks)}")

    # Stream + batch embed and insert into Chroma
    BATCH_SIZE = embed_batch_size
    total = len(final_chunks)

    for i in range(0, total, BATCH_SIZE):
        batch_docs = final_chunks[i:i+BATCH_SIZE]
        batch_meta = final_metadata[i:i+BATCH_SIZE]
        batch_ids = [str(uuid.uuid4()) for _ in batch_docs]

        try:
            batch_embeddings = embed_texts(batch_docs)
            collection.add(
                documents=batch_docs,
                embeddings=batch_embeddings,
                metadatas=batch_meta,
                ids=batch_ids
            )
            logging.info(f"✅ Inserted batch {i//BATCH_SIZE + 1} ({i}/{total})")
        except Exception as e:
            logging.error(f"❌ Embedding or insert failed for batch starting at index {i}: {e}")

# Process dataset in chunks of 10,000 rows
BATCH_SIZE = 10000
num_batches = len(dataset) // BATCH_SIZE + (1 if len(dataset) % BATCH_SIZE > 0 else 0)

for i in range(num_batches):
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, len(dataset))

    logging.info(f"🔁 Processing batch {i+1}/{num_batches} ({start_idx} - {end_idx})")
    dataset_slice = dataset.select(range(start_idx, end_idx))
    process_and_insert(dataset_slice)

logging.info("🎉 Script completed successfully.")