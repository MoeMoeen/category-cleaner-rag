# app/schemas.py
from typing import List, Dict
from pydantic import BaseModel

class CategoryRequest(BaseModel):
    """
    Input schema: user provides a list of messy product categories.
    Example:
    {
      "categories": ["Jeans", "tshirt", "Shoes", "Shorts"]
    }
    """
    categories: List[str]

class CategoryResponse(BaseModel):
    """
    Output schema: mapping from Shopify taxonomy full paths
    to the list of input categories that belong to that path.
    Example:
    {
      "mapping": {
        "Apparel & Accessories > Clothing > Jeans": ["Jeans"],
        "Apparel & Accessories > Clothing > Shirts & Tops > T-Shirts": ["tshirt"],
        "Apparel & Accessories > Shoes": ["Shoes"],
        "Apparel & Accessories > Clothing > Shorts": ["Shorts"]
      }
    }
    """
    mapping: Dict[str, List[str]]
