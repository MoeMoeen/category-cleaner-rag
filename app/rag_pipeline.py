# app/rag_pipeline.py
"""
RAG (Retrieval-Augmented Generation) pipeline for product category cleaning.

This module handles:
1. Embedding Shopify taxonomy full paths
2. Building & saving FAISS index
3. Retrieving candidate categories for a given input
4. Passing candidates + input to LLM for classification
"""

import os
import pickle
import numpy as np
from numpy.typing import NDArray
import faiss
import pandas as pd
from pathlib import Path
from openai import OpenAI

from app.config import OPENAI_API_KEY, EMBED_MODEL, LLM_MODEL, TAXONOMY_PATH, get_logger
from app.taxonomy_loader import load_taxonomy

# -----------------------------------------------------
# 🔹 Setup logging
# -----------------------------------------------------
logger = get_logger()

# -----------------------------------------------------
# 🔹 Initialize OpenAI client
# -----------------------------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------------------------
# 🔹 Globals (index + df)
# -----------------------------------------------------
INDEX_PATH = Path("data/faiss_index.bin")
DF_PATH = Path("data/taxonomy_df.pkl")

index = None
df: pd.DataFrame | None = None


# -----------------------------------------------------
# 🔹 Embeddings
# -----------------------------------------------------
def embed_text(text: str) -> list[float]:
    """
    Create an embedding vector for a given text using OpenAI.
    """
    resp = client.embeddings.create(
        input=text,
        model=EMBED_MODEL
    )
    return resp.data[0].embedding


def build_index(force: bool = False):
    """
    Build FAISS index from Shopify taxonomy full paths.

    Steps:
    1. Load taxonomy DataFrame
    2. Generate embeddings
    3. Save FAISS index + DataFrame for reuse
    """
    global index, df

    if INDEX_PATH.exists() and DF_PATH.exists() and not force:
        logger.info("Loading existing FAISS index and taxonomy DataFrame...")
        index = faiss.read_index(str(INDEX_PATH))
        df = pickle.loads(DF_PATH.read_bytes())
        return

    logger.info("Building new FAISS index...")
    df = load_taxonomy(TAXONOMY_PATH)

    # Generate embeddings
    embeddings = []
    for text in df["full_path"]:
        emb = embed_text(text)
        embeddings.append(emb)

    embedding_matrix: NDArray[np.float32] = np.array(embeddings).astype("float32")

    # Create FAISS index
    dim = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dim)
    # add vectors to the index 
    index.add(embedding_matrix) # type: ignore 

    # Save index + dataframe
    faiss.write_index(index, str(INDEX_PATH))
    DF_PATH.write_bytes(pickle.dumps(df))

    logger.info(f"FAISS index built with {index.ntotal} entries")


def search_taxonomy(query: str, k: int = 5) -> pd.DataFrame:
    """
    Search Shopify taxonomy for the closest matches to a query.

    Returns a DataFrame with top-k candidates.
    """
    global index, df

    if index is None or df is None:
        logger.info("Index not loaded — building/loading now...")
        build_index()
    # Ensure index and df are loaded after build_index
    if index is None or df is None:
        raise RuntimeError("FAISS index or taxonomy DataFrame could not be loaded.")

    # Embed query
    q_emb : NDArray[np.float32] = np.array([embed_text(query)], dtype="float32")

    # Perform search
    D, I = index.search(q_emb, k)  # D = distances, I = indices # type: ignore
    results = df.iloc[I[0]].copy()
    results["distance"] = D[0]

    logger.info(f"Search results for '{query}':\n{results[['full_path','distance']].to_string(index=False)}")
    return results


def classify_with_llm(query: str, candidates: pd.DataFrame) -> str:
    """
    Use an LLM to pick the best taxonomy path from retrieved candidates.
    """
    prompt = f"""
    You are given a messy product category: "{query}".

    Candidate Shopify taxonomy categories are:
    {chr(10).join([f"- {c}" for c in candidates['full_path']])}

    Choose the single best matching category and return ONLY its path.
    """

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a precise product categorization engine."},
            {"role": "user", "content": prompt}
        ]
    )
    content = resp.choices[0].message.content
    choice = content.strip() if content is not None else ""
    logger.info(f"LLM classified '{query}' as: {choice}")
    return choice


def categorize_inputs(inputs: list[str]) -> dict[str, list[str]]:
    """
    Full pipeline: for each input category:
      1. Search taxonomy for candidates
      2. Ask LLM to choose the best
      3. Aggregate results in {taxonomy_path: [inputs]} mapping
    """
    mapping: dict[str, list[str]] = {}

    for raw in inputs:
        candidates = search_taxonomy(raw, k=5)
        best = classify_with_llm(raw, candidates)

        if best not in mapping:
            mapping[best] = []
        mapping[best].append(raw)

    return mapping


if __name__ == "__main__":
    # Simple manual test runner for this module
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="RAG pipeline smoke test")
    parser.add_argument("--query", type=str, default="shoes", help="Query/category to search")
    parser.add_argument("--k", type=int, default=5, help="Top-k candidates to retrieve")
    parser.add_argument("--force", action="store_true", help="Force rebuild of FAISS index")
    parser.add_argument("--classify", action="store_true", help="Also run LLM classification on the retrieved candidates")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not set; cannot run embeddings/LLM.")
        sys.exit(1)

    try:
        build_index(force=args.force)
        print(f"Index ready. Searching for: '{args.query}' (k={args.k})\n")
        res = search_taxonomy(args.query, k=args.k)
        cols = [c for c in ["full_path", "distance"] if c in res.columns]
        print(res[cols].to_string(index=False))

        if args.classify:
            choice = classify_with_llm(args.query, res)
            print(f"\nLLM choice: {choice}")
    except Exception as e:
        print(f"Error during test run: {e}")
        sys.exit(1)


# python rag_pipeline.py --query "women running shoes" --k 5
# python rag_pipeline.py --query "baby stroller" --k 10 --force
# python rag_pipeline.py --query "laptop bag" --classify