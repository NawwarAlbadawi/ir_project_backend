"""
=============================================================================
 Preprocessing Service — main.py
 Port: 8001
=============================================================================
 Responsibilities:
   - Text normalization, tokenization, stop-word removal, stemming,
     lemmatization for both documents and queries.
   - Loading and caching datasets from `ir_datasets`.
   - Exposing batch-preprocessing endpoints for other services to consume.
=============================================================================
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware

from preprocessing_service.schemas import (
    TextRequest,
    BatchTextRequest,
    PreprocessOptions,
    TextResponse,
    BatchTextResponse,
    DatasetStatusResponse,
)
from preprocessing_service.preprocessor import Preprocessor
from preprocessing_service.dataset_loader import DatasetLoader

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("preprocessing_service")

# ---------------------------------------------------------------------------
# Shared state (in-memory singletons)
# ---------------------------------------------------------------------------
preprocessor: Preprocessor = None
dataset_loader: DatasetLoader = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy objects once at startup."""
    global preprocessor, dataset_loader
    logger.info("Starting Preprocessing Service — downloading NLTK data if needed…")
    preprocessor = Preprocessor()
    dataset_loader = DatasetLoader(preprocessor=preprocessor)
    logger.info("Preprocessing Service ready.")
    yield
    logger.info("Preprocessing Service shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Preprocessing Service",
    description=(
        "Handles text normalization, stemming, lemmatization, "
        "and dataset loading / caching for the IR system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Health
# ===========================================================================
@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "service": "preprocessing"}


# ===========================================================================
# Text preprocessing endpoints
# ===========================================================================
@app.post("/preprocess/text", response_model=TextResponse, tags=["preprocessing"])
def preprocess_text(request: TextRequest):
    """
    Preprocess a **single** text string.

    Returns both the original text and the cleaned/processed version,
    plus the list of individual tokens.
    """
    tokens = preprocessor.preprocess(
        text=request.text,
        options=request.options or PreprocessOptions(),
    )
    return TextResponse(
        original=request.text,
        tokens=tokens,
        cleaned=" ".join(tokens),
    )


@app.post("/preprocess/batch", response_model=BatchTextResponse, tags=["preprocessing"])
def preprocess_batch(request: BatchTextRequest):
    """
    Preprocess a **list** of text strings in one call.

    Useful for other services that need to preprocess large batches
    without making individual HTTP calls per document.
    """
    results: list[TextResponse] = []
    opts = request.options or PreprocessOptions()
    for text in request.texts:
        tokens = preprocessor.preprocess(text=text, options=opts)
        results.append(
            TextResponse(
                original=text,
                tokens=tokens,
                cleaned=" ".join(tokens),
            )
        )
    return BatchTextResponse(results=results)


# ===========================================================================
# Dataset endpoints
# ===========================================================================
@app.post("/dataset/load", tags=["dataset"])
def load_dataset(
    dataset_name: str = Query(..., description="e.g. 'quora' or 'msmarco'"),
    background_tasks: BackgroundTasks = None,
):
    """
    Trigger loading + preprocessing of a full dataset.

    This is an **asynchronous** operation — it runs in a background thread
    and can take several minutes for large corpora.
    Use ``GET /dataset/status`` to poll progress.
    """
    if dataset_loader.is_loading(dataset_name):
        return {"message": f"Dataset '{dataset_name}' is already being loaded."}
    background_tasks.add_task(dataset_loader.load_dataset, dataset_name)
    return {"message": f"Loading of '{dataset_name}' started in background."}


@app.get("/dataset/status", response_model=DatasetStatusResponse, tags=["dataset"])
def dataset_status(
    dataset_name: str = Query(..., description="e.g. 'quora' or 'msmarco'"),
):
    """Return the current load/preprocess status of a dataset."""
    return dataset_loader.get_status(dataset_name)


@app.get("/dataset/docs", tags=["dataset"])
def get_documents(
    dataset_name: str = Query(...),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=10_000),
):
    """
    Return a page of **preprocessed** documents from a cached dataset.

    Each document is a dict: ``{doc_id, tokens, cleaned_text}``.
    """
    docs = dataset_loader.get_documents(dataset_name, offset=offset, limit=limit)
    if docs is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not loaded. Call POST /dataset/load first.",
        )
    return {"dataset": dataset_name, "offset": offset, "count": len(docs), "documents": docs}


@app.get("/dataset/queries", tags=["dataset"])
def get_queries(
    dataset_name: str = Query(...),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=10_000),
):
    """Return a page of preprocessed queries from the cached dataset."""
    queries = dataset_loader.get_queries(dataset_name, offset=offset, limit=limit)
    if queries is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not loaded.",
        )
    return {"dataset": dataset_name, "offset": offset, "count": len(queries), "queries": queries}


@app.get("/dataset/qrels", tags=["dataset"])
def get_qrels(dataset_name: str = Query(...)):
    """
    Return the relevance judgments (qrels) for a dataset.

    Format: ``{ query_id: { doc_id: relevance_score } }``
    """
    qrels = dataset_loader.get_qrels(dataset_name)
    if qrels is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not loaded.",
        )
    return {"dataset": dataset_name, "qrels": qrels}


@app.get("/dataset/raw_docs", tags=["dataset"])
def get_raw_documents(
    dataset_name: str = Query(...),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=10_000),
):
    """
    Return raw (un-preprocessed) documents for display purposes in the UI.

    Format: ``{doc_id, text}``
    """
    docs = dataset_loader.get_raw_documents(dataset_name, offset=offset, limit=limit)
    if docs is None:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' not loaded.",
        )
    return {"dataset": dataset_name, "offset": offset, "count": len(docs), "documents": docs}


@app.get("/dataset/all_doc_ids", tags=["dataset"])
def get_all_doc_ids(dataset_name: str = Query(...)):
    """Return the complete ordered list of document IDs for a loaded dataset."""
    doc_ids = dataset_loader.get_all_doc_ids(dataset_name)
    if doc_ids is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not loaded.")
    return {"dataset": dataset_name, "count": len(doc_ids), "doc_ids": doc_ids}