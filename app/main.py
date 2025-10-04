# app/main.py
from fastapi import FastAPI
from app.schemas import CategoryRequest, CategoryResponse
from app.rag_pipeline import categorize_inputs
from app.config import get_logger
from contextlib import asynccontextmanager
from app.rag_pipeline_v2 import init_pipeline, close_pipeline

logger = get_logger()

@asynccontextmanager
async def lifespan(app):
    # ---- STARTUP ----
    # Load or rebuild your FAISS index + taxonomy
    init_pipeline(force_rebuild=False)
    print("✅ RAG pipeline initialized")

    # Hand over control to the app (requests are served now)
    yield

    # ---- SHUTDOWN ----
    # Do cleanup here (close DB, release FAISS, etc.)
    close_pipeline()
    print("🛑 RAG pipeline resources released")

app = FastAPI(title="Product Category Cleaner", lifespan=lifespan)

@app.get("/")
def root():
    logger.info("Root endpoint accessed")
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
    logger.info(f"Categorization result: {result}")
    return {"mapping": result}


