# app/config.py
import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# OpenAI settings
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL: str = os.getenv("OPENAI_LLM_MODEL", "gpt-5")

# Paths
TAXONOMY_PATH: str = os.getenv("TAXONOMY_PATH", "data/shopify_taxonomy.json")
