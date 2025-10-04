# app/taxonomy_loader.py
import json
from pathlib import Path
import pandas as pd
from typing import Any, Dict, List


def load_taxonomy(path: str | Path) -> pd.DataFrame:
    """
    Load Shopify taxonomy JSON from `dist/taxonomy.json` (or similar)
    and return a DataFrame with:
      - id
      - name
      - level
      - full_path (from 'full_name' field if present)
      - parent_id
      - vertical (top-level vertical name)
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []

    for vertical in data.get("verticals", []):
        vertical_name = vertical.get("name", "UNKNOWN")

        for category in vertical.get("categories", []):
            rows.append({
                "id": category.get("id"),
                "name": category.get("name"),
                "level": category.get("level"),
                "full_path": category.get("full_name") or category.get("name"),
                "parent_id": category.get("parent_id"),
                "vertical": vertical_name,
            })

    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    return df


if __name__ == "__main__":
    # Quick test: run this file directly to preview
    taxonomy_path = "product-taxonomy/dist/en/categories.json"  # adjust path if needed
    df = load_taxonomy(taxonomy_path)
    print(df.head(20))
    print("Total categories:", len(df))
