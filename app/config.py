# app/config.py
import os
from dotenv import load_dotenv
import logging
from pathlib import Path

# Load .env file if present
load_dotenv()

# OpenAI settings
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-5")

# Paths
TAXONOMY_PATH: str = os.getenv("TAXONOMY_PATH", "product-taxonomy/dist/en/categories.json")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_FILENAME = os.getenv("INDEX_FILENAME", "faiss_index.bin")
DF_FILENAME = os.getenv("DF_FILENAME", "taxonomy_df.parquet")
META_FILENAME = os.getenv("META_FILENAME", "faiss_index.meta.json")

# RAG knobs
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))  # batching embeddings
DEFAULT_K = int(os.getenv("DEFAULT_K", "5"))                   # top-k
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))              # cosine score threshold for NO_MATCH
NO_MATCH_LABEL = os.getenv("NO_MATCH_LABEL", "__NO_MATCH__")   # sentinel for low-confidence results



# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)
logger.info(f"Configuration loaded: EMBED_MODEL={EMBED_MODEL}, LLM_MODEL={LLM_MODEL}, TAXONOMY_PATH={TAXONOMY_PATH}")

def get_logger() -> logging.Logger:
    return logging.getLogger(__name__)