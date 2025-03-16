# embed/korean_law_open_data_precedents.py

import json
import uuid
import logging
from scripts.embed.embedding_model import model, collection
from datasets import load_dataset

# Setup logging
logging.basicConfig(
    filename="embed_store.log",
    level=logging.INFO,  # Change to DEBUG for more details
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Script started: korean_law_open_data_precedents.py")

data = load_dataset("joonhok-exo-ai/korean_law_open_data_precedents")

FIELDS_OF_INTEREST = [
    "판례정보일련번호",
    "사건명",
    "사건번호",
    "선고일자",
    "선고",
    "법원명",
    "사건종류명",
    "판결유형",
    "판시사항",
    "판결요지",
    "참조조문",
    "참조판례",
    "판결유형",
    "전문",
]