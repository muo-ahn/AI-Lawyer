# embed/embed_store.py

import json
import uuid
import logging
from embedding_model import embed_texts, collection
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename="embed_store.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Script started: embed_store.py")

file_path = os.getenv("FILE1")
try:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info(f"Successfully loaded JSON file: {file_path}")
except Exception as e:
    logging.error(f"Failed to load JSON file: {e}")
    raise

FIELDS_OF_INTEREST = [
    "http://www.aihub.or.kr/kb/law/caseNumber",
    "http://www.aihub.or.kr/kb/law/caseName",
    "http://www.aihub.or.kr/kb/law/caseType",
    "http://www.aihub.or.kr/kb/law/courtName",
    "http://www.aihub.or.kr/kb/law/sentenceDate",
    "http://www.aihub.or.kr/kb/law/judgementAbstract",
    "http://www.aihub.or.kr/kb/law/precedentText",
    "http://www.aihub.or.kr/kb/law/judgementNote",
]

text_chunks = []
metadata_list = []

for uri, properties in data.items():
    combined_text = []
    meta = {"uri": uri}

    for field in FIELDS_OF_INTEREST:
        if field in properties:
            field_values = properties[field]
            extracted_values = [fv["value"] for fv in field_values if "value" in fv]
            field_text = " ".join(extracted_values)
            combined_text.append(field_text)

            if field == "http://www.aihub.or.kr/kb/law/caseNumber":
                meta["caseNumber"] = extracted_values[0] if extracted_values else ""

    if combined_text:
        text_chunk = "\n".join(combined_text)
        text_chunks.append(text_chunk)
        metadata_list.append(meta)

logging.info(f"Extracted {len(text_chunks)} text chunks.")

def chunk_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

final_chunks = []
final_metadata = []

for t, m in zip(text_chunks, metadata_list):
    if len(t) > 1000:
        for i, sub_t in enumerate(chunk_text(t)):
            meta = dict(m)
            meta["chunk_index"] = i
            final_chunks.append(sub_t)
            final_metadata.append(meta)
    else:
        final_chunks.append(t)
        final_metadata.append(m)

logging.info(f"Total chunks after splitting: {len(final_chunks)}")

# ✅ Process and insert in batches
BATCH_SIZE = 64
total = len(final_chunks)

try:
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
            logging.info(f"Inserted batch {i//BATCH_SIZE + 1} ({i}/{total})")
        except Exception as e:
            logging.error(f"Embedding or insert failed for batch starting at index {i}: {e}")

    logging.info("Successfully inserted all batches.")
except Exception as e:
    logging.error(f"Outer embedding loop failed: {e}")
    raise
