
### We’ll set up a FastAPI service that wraps the RAG pipeline we designed. It’ll take a list of messy categories (["Jeans", "tshirt", "tees", "shoes", ...]), run them through embeddings + retrieval + LLM, and return a JSON mapping:

{
  "Apparel & Accessories > Clothing > Jeans": ["Jeans"],
  "Apparel & Accessories > Clothing > Shirts & Tops > T-Shirts": ["tshirt", "tees", "TeeShirts"],
  "Apparel & Accessories > Shoes > Sandals": ["sandals"],
  "Apparel & Accessories > Clothing > Swimwear > Bikinis": ["bikini"],
  "Apparel & Accessories > Clothing > Shorts": ["short", "Shorts"]
}


product_category_cleaner/
│
├── app/                       # all application code
│   ├── __init__.py
│   ├── main.py                 # FastAPI entrypoint + routes
│   ├── config.py               # settings, env vars
│   ├── schemas.py              # Pydantic request/response models
│   ├── rag_pipeline.py         # embeddings, FAISS, retrieval, LLM
│   ├── taxonomy_loader.py      # load + flatten Shopify taxonomy JSON
│   └── parse_taxonomy.py       # CLI/utility script (can reuse taxonomy_loader)
│
├── data/                       # raw data (taxonomy.json lives here)
│   └── .gitkeep
│
├── tests/                      # pytest tests
│   └── test_pipeline.py
│
├── .env                        # secrets
├── .gitignore
├── pyproject.toml              # uv project file
└── README.md


### Flow (end-to-end)

1. Load taxonomy JSON (taxonomy_loader.py) → flatten into DataFrame with id, name, full_path.

2. Precompute embeddings for all full_path rows (script or at service startup). Store in FAISS index.

3. API request → user posts a list of messy categories.

4. Pipeline (rag_pipeline.py):

- For each input: create embedding, retrieve top-N taxonomy candidates.
- Pass input + candidates into LLM → get best full path.
- Collect results.

5. Return JSON with keys = Shopify full paths, values = list of original inputs that map there.


### API contract (example)

Request:

{
  "categories": ["Jeans", "tshirt", "tees", "shoes", "sandals", "TeeShirts", "bikini", "short", "Shorts"]
}

Response:

{
  "Apparel & Accessories > Clothing > Jeans": ["Jeans"],
  "Apparel & Accessories > Clothing > Shirts & Tops > T-Shirts": ["tshirt", "tees", "TeeShirts"],
  "Apparel & Accessories > Shoes > Sandals": ["sandals"],
  "Apparel & Accessories > Clothing > Swimwear > Bikinis": ["bikini"],
  "Apparel & Accessories > Clothing > Shorts": ["short", "Shorts"]
}


### Clarifying taxonomy_loader.py vs parse_taxonomy.py

taxonomy_loader.py
→ reusable library code: functions to load and flatten taxonomy JSON into a DataFrame.
→ imported inside the API / pipeline.

parse_taxonomy.py
→ a standalone script for manual inspection/debugging.
→ uses functions from taxonomy_loader.py.
→ e.g. uv run python app/parse_taxonomy.py prints the first 20 categories, depths, etc.