# app/main.py
from fastapi import FastAPI
from app.schemas import CategoryRequest, CategoryResponse
from app.rag_pipeline import categorize_inputs

app = FastAPI(title="Product Category Cleaner")

@app.get("/")
def root():
    return {"message": "Product Category Cleaner API is running"}

@app.post("/categorize", response_model=CategoryResponse)
def categorize(request: CategoryRequest):
    """
    Accepts a list of messy product categories,
    runs them through the RAG pipeline,
    and returns a mapping of Shopify taxonomy full paths
    to the original inputs that belong there.
    """
    result = categorize_inputs(request.categories)
    return {"mapping": result}
