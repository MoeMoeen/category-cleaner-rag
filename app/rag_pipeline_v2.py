# app/rag_pipeline_v2.py
"""
RAG pipeline for product category cleaning (production-ready version).

Implements:
- Robust taxonomy loading + embedding build with batching
- Cosine similarity via FAISS IndexFlatIP (unit-normalized vectors)
- Safe persistence (Parquet + metadata with taxonomy hash & embed model)
- Retry/backoff for OpenAI calls
- Clamp k, score threshold + NO_MATCH sentinel
- Deterministic LLM selection (JSON index output)
- Concurrency safety and explicit warmup
- Structured logging and timings
"""

from __future__ import annotations

import json
import pickle
import logging
import random
import threading
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, List, Tuple

import faiss
import numpy as np
import openai
import pandas as pd
from rapidfuzz import fuzz, process

from app.config import (
    DATA_DIR,
    INDEX_FILENAME,
    DF_FILENAME,
    META_FILENAME,
    EMBED_BATCH_SIZE,
    DEFAULT_K,
    MIN_SCORE,
    NO_MATCH_LABEL,
    EMBED_MODEL,
    LLM_MODEL,
    OPENAI_API_KEY,
    TAXONOMY_PATH,
)
from app.taxonomy_loader import load_taxonomy

# ----------------------------- Logging ---------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("rag_pipeline")

# ----------------------------- Types -----------------------------------------
IndexType = faiss.Index | None

# ----------------------------- Globals ---------------------------------------
_client: Any = None
_index: IndexType = None
_df: pd.DataFrame | None = None
_lock = threading.RLock()

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = DATA_DIR / INDEX_FILENAME
DF_PATH = DATA_DIR / DF_FILENAME
META_PATH = DATA_DIR / META_FILENAME

# ----------------------------- Utilities -------------------------------------
def _require_openai() -> Any:
    """Ensure OpenAI is configured and return a client."""
    global _client
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Set it in your .env before running.")
    if _client is None:
        _client = openai.OpenAI()  # uses env var # type: ignore[reportGeneralTypeIssues]
    return _client


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_text(s: str) -> str:
    """Light normalization for embeddings/retrieval consistency."""
    return " ".join(s.lower().strip().split())


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Unit-normalize vectors row-wise for cosine/IP indexing."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def _retry(fn, *, tries=5, base_delay=0.5, max_delay=8.0, jitter=0.1, what=""):
    """Simple retry wrapper with exponential backoff + jitter."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == tries:
                logger.error(f"{what} failed after {tries} attempts: {e}")
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, jitter)
            logger.warning(f"{what} failed (attempt {attempt}/{tries}): {e} → retrying in {delay:.2f}s")
            time.sleep(delay)

# ----------------------------- Persistence -----------------------------------
@dataclass
class Meta:
    embed_model: str
    index_type: str
    vector_dim: int
    ntotal: int
    taxonomy_sha256: str
    taxonomy_path: str
    built_at: float

    def to_json(self) -> dict:
        return {
            "embed_model": self.embed_model,
            "index_type": self.index_type,
            "vector_dim": self.vector_dim,
            "ntotal": self.ntotal,
            "taxonomy_sha256": self.taxonomy_sha256,
            "taxonomy_path": self.taxonomy_path,
            "built_at": self.built_at,
        }

    @staticmethod
    def from_path(p: Path) -> "Meta | None":
        if not p.exists():
            return None
        return Meta(**json.loads(p.read_text(encoding="utf-8")))


def _save_meta(meta: Meta) -> None:
    META_PATH.write_text(json.dumps(meta.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")

# ----------------------------- Embeddings ------------------------------------
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Batch-embed a list of texts with retries."""
    client = _require_openai()
    t0 = time.perf_counter()

    def _call():
        return client.embeddings.create(input=texts, model=EMBED_MODEL)

    resp: Any = _retry(_call, what="OpenAI embeddings")
    vecs = [d.embedding for d in resp.data]  # type: ignore[attr-defined]

    dur = (time.perf_counter() - t0) * 1000
    logger.info(f"Embedded {len(texts)} texts in {dur:.1f} ms (model={EMBED_MODEL})")
    return vecs

# ----------------------------- Build Index / Load ----------------------------------
def build_index(force: bool = False) -> None:
    """Build or load FAISS index + taxonomy DataFrame (thread-safe)."""
    global _index, _df

    with _lock:
        if not force and INDEX_PATH.exists() and DF_PATH.exists() and META_PATH.exists():
            meta = Meta.from_path(META_PATH)
            tax_hash = _file_sha256(Path(TAXONOMY_PATH))
            if meta and meta.embed_model == EMBED_MODEL and meta.taxonomy_sha256 == tax_hash:
                logger.info("Loading FAISS index + DataFrame from disk...")
                _index = faiss.read_index(str(INDEX_PATH))
                # Load dataframe via pickle
                _df = pickle.loads(DF_PATH.read_bytes())
                # Guard for type checker
                assert _index is not None
                assert _df is not None
                logger.info(f"Loaded index with ntotal={_index.ntotal}, rows={len(_df)}")
                return
            else:
                logger.info("Metadata mismatch (model or taxonomy changed) → rebuilding index.")

        logger.info("Building taxonomy DataFrame...")
        _df = load_taxonomy(TAXONOMY_PATH)

        # Keep original path for output, create normalized version for embedding
        _df["full_path_norm"] = _df["full_path"].map(_normalize_text)

        # Batch embeddings
        texts = _df["full_path_norm"].tolist()
        vecs: List[List[float]] = []
        batch = EMBED_BATCH_SIZE
        t0 = time.perf_counter()
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            vecs.extend(embed_texts(chunk))
        embed_ms = (time.perf_counter() - t0) * 1000
        logger.info(f"Total embedding time for {len(texts)} items: {embed_ms:.1f} ms")

        # Build FAISS cosine index (IndexFlatIP with unit vectors)
        embeddings_matrix = np.array(vecs, dtype="float32")
        embeddings_matrix = _l2_normalize(embeddings_matrix)
        dim = embeddings_matrix.shape[1]
        idx = faiss.IndexFlatIP(dim)
        idx.add(embeddings_matrix)  # type: ignore[reportGeneralTypeIssues]
        _index = idx

        # Persist index + DF + metadata
        faiss.write_index(_index, str(INDEX_PATH))
        # Persist dataframe via pickle
        DF_PATH.write_bytes(pickle.dumps(_df))
        meta = Meta(
            embed_model=EMBED_MODEL,
            index_type="IndexFlatIP",
            vector_dim=dim,
            ntotal=_index.ntotal,
            taxonomy_sha256=_file_sha256(Path(TAXONOMY_PATH)),
            taxonomy_path=str(TAXONOMY_PATH),
            built_at=time.time(),
        )
        _save_meta(meta)
        # free memory
        del embeddings_matrix, vecs

        logger.info(f"Built FAISS index (ntotal={_index.ntotal}, dim={dim}) and saved artifacts to {DATA_DIR}/")

# ----------------------------- Public Init / Close ---------------------------
def init_pipeline(force_rebuild: bool = False) -> None:
    """Public warmup function; call on app startup."""
    _ = _require_openai()  # early credential check
    build_index(force=force_rebuild)

def close_pipeline() -> None:
    """Public cleanup function; call on app shutdown."""
    global _client, _index, _df
    with _lock:
        _client = None
        _index = None
        _df = None
    logger.info("RAG pipeline resources released.")

# ----------------------------- Search ----------------------------------------
def _search(query: str, k: int) -> pd.DataFrame:
    """Internal search: normalized embedding → FAISS → top-k DataFrame with scores."""
    global _index, _df
    if _index is None or _df is None:
        build_index()
    # Ensure loaded after build
    if _index is None or _df is None:
        raise RuntimeError("Index or DataFrame not initialized after build.")

    # Clamp k
    assert _index is not None
    k = max(1, min(k, _index.ntotal))

    # Embed + normalize
    t0 = time.perf_counter()
    q_vec = np.array(embed_texts([_normalize_text(query)]), dtype="float32")
    q_vec = _l2_normalize(q_vec)
    embed_ms = (time.perf_counter() - t0) * 1000

    # Search
    t1 = time.perf_counter()
    scores, idxs = _index.search(q_vec, k)  # type: ignore[reportGeneralTypeIssues]  # cosine scores ∈ [-1, 1]
    search_ms = (time.perf_counter() - t1) * 1000

    assert _df is not None
    rows = _df.iloc[idxs[0]].copy()
    rows["score"] = scores[0]  # higher is better
    rows["rank"] = np.arange(1, len(rows) + 1)

    logger.info(
        f"search(query='{query}', k={k}) → top1={rows.iloc[0]['full_path']} "
        f"score={rows.iloc[0]['score']:.3f} | embed={embed_ms:.1f}ms search={search_ms:.1f}ms"
    )
    return rows

def search_taxonomy(query: str, k: int = DEFAULT_K) -> pd.DataFrame:
    """Public search with safe defaults."""
    return _search(query, k)

# ----------------------------- LLM selection ---------------------------------
def _classify_with_llm(query: str, candidates: pd.DataFrame) -> int | None:
    """
    Ask the LLM to choose exactly one candidate by **index**.

    Returns:
        int index (0-based) or None on parse/validation failure.
    """
    client = _require_openai()

    # prepare numbered panel + scores to help model
    panel = []
    for i, row in enumerate(candidates.itertuples(index=False)):
        panel.append(f"[{i}] {row.full_path}  (score={row.score:.3f})")
    panel_text = "\n".join(panel)

    system = "You are a precise product categorization engine. Be deterministic."
    user = f"""
Given the messy category: "{query}"

Choose exactly ONE best match from the numbered candidates below.
Return ONLY a JSON object with an integer field "index" (0-based).

Candidates:
{panel_text}

Constraints:
- Respond with JSON like: {{ "index": 2 }}
- The index MUST be one of the listed candidates.
- No explanations. Temperature=0.
"""

    t0 = time.perf_counter()
    def _call():
        return client.chat.completions.create(
            model=LLM_MODEL,
            temperature=1,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}]
        )
    resp: Any = _retry(_call, what="OpenAI classification")
    llm_ms = (time.perf_counter() - t0) * 1000

    text = resp.choices[0].message.content.strip()  # type: ignore[attr-defined]
    try:
        data = json.loads(text)
        idx = int(data["index"])
    except Exception:
        logger.warning(f"LLM output not valid JSON index: {text!r} (took {llm_ms:.1f}ms)")
        return None

    if 0 <= idx < len(candidates):
        logger.info(f"LLM chose index {idx} (took {llm_ms:.1f}ms)")
        return idx

    logger.warning(f"LLM returned out-of-range index {idx} (took {llm_ms:.1f}ms)")
    return None

# ----------------------------- Public API ------------------------------------
def categorize_inputs(
    inputs: List[str],
    k: int = DEFAULT_K,
    min_score: float = MIN_SCORE,
    return_debug: bool = False,
) -> dict[str, List[str]] | Tuple[dict[str, List[str]], dict[str, dict]]:
    """
    For each input string:
      1) retrieve top-k candidates (cosine scores)
      2) if top-1 score < min_score → return NO_MATCH sentinel
      3) else ask LLM for a single candidate index (JSON). If invalid → fallback to top-1
      4) aggregate mapping: {selected_full_path: [original_inputs...]}

    Args:
        inputs: list of raw category strings.
        k: retrieval depth (clamped to index size).
        min_score: cosine threshold to avoid bad matches.
        return_debug: if True, also return per-input debug info.

    Returns:
        mapping OR (mapping, debug_map)
    """
    results: dict[str, List[str]] = {}
    debug: dict[str, dict] = {}

    for raw in inputs:
        cands = search_taxonomy(raw, k=k)
        top_score = float(cands.iloc[0]["score"])
        if top_score < min_score:
            chosen = NO_MATCH_LABEL
        else:
            idx = _classify_with_llm(raw, cands)
            if idx is None:
                # Fallback 1: top-1 retrieval
                chosen = str(cands.iloc[0]["full_path"])
                # Fallback 2 (optional): if retrieval close, try fuzzy between raw and candidates
                # Keep it simple: only if top two within 0.02, pick better fuzz ratio
                if len(cands) > 1 and abs(cands.iloc[0]["score"] - cands.iloc[1]["score"]) < 0.02:
                    choices = cands["full_path"].tolist()
                    fuzz_best = process.extractOne(raw, choices, scorer=fuzz.WRatio)
                    if fuzz_best and fuzz_best[0]:
                        chosen = fuzz_best[0]
            else:
                chosen = str(cands.iloc[idx]["full_path"])

        results.setdefault(chosen, []).append(raw)

        if return_debug:
            debug[raw] = {
                "top_k": cands[["full_path", "score", "rank"]].to_dict(orient="records"),
                "top1_score": top_score,
                "min_score": min_score,
                "chosen": chosen,
                "no_match": chosen == NO_MATCH_LABEL,
            }

    return (results, debug) if return_debug else results


if __name__ == "__main__":
    # Simple manual test runner for v2 pipeline
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="RAG v2 pipeline smoke test")
    parser.add_argument("--queries", nargs="+", type=str, required=True, help="One or more queries")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Top-k candidates to retrieve")
    parser.add_argument(
        "--force-rebuild", action="store_true", help="Force rebuild of FAISS index and embeddings"
    )
    parser.add_argument("--classify", action="store_true", help="Also run LLM classification on the retrieved candidates")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE, help="Cosine threshold for NO_MATCH")
    parser.add_argument("--debug", action="store_true", help="Return and print debug info for classification")
    args = parser.parse_args()

    try:
        init_pipeline(force_rebuild=args.force_rebuild)
        print(f"Index ready. Searching for: {args.queries} (k={args.k})\n")
        cols = ["full_path", "score", "rank"]
        for q in args.queries:
            print(f"\n=== Query: {q} ===")
            res = search_taxonomy(q, k=args.k)
            use_cols = [c for c in cols if c in res.columns]
            print(res[use_cols].to_string(index=False))

        if args.classify:
            if args.debug:
                mapping, dbg = categorize_inputs(args.queries, k=args.k, min_score=args.min_score, return_debug=True)
                print(f"\nMapping: {mapping}")
                print(f"Debug: {json.dumps(dbg, indent=2)}")
            else:
                mapping = categorize_inputs(args.queries, k=args.k, min_score=args.min_score)
                print(f"\nMapping: {mapping}")
    except Exception as e:
        print(f"Error during v2 test run: {e}")
        sys.exit(1)


# python rag_pipeline_v2.py --query "women running shoes" --k 5
# python rag_pipeline_v2.py --query "laptop bag" --classify --debug
# python rag_pipeline_v2.py --query "stroller" --k 8 --force-rebuild