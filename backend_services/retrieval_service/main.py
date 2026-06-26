"""
=============================================================================
 Retrieval Service — main.py
 Port: 8003
=============================================================================
 Loads indexes from disk (built by the Indexing Service) and exposes
 a unified /retrieve endpoint that dispatches to the chosen model.
=============================================================================
"""

import logging
import time
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from retrieval_service.schemas import RetrievalRequest, RetrievalResponse, RetrievalResult
from retrieval_service.hybrid_retriever import serial_hybrid, parallel_hybrid

# Reuse indexer objects from the indexing_service package
# (They live in the same venv, just different process conceptually)
# In a real microservice deployment, you'd load from shared disk or a model server.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexing_service.inverted_index import InvertedIndex
from indexing_service.tfidf_indexer import TFIDFIndexer
from indexing_service.bm25_indexer import BM25Indexer
from indexing_service.embedding_indexer import Word2VecIndexer, BERTIndexer

logger = logging.getLogger("retrieval_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Per-dataset loaded indexes
# ---------------------------------------------------------------------------
class _DatasetIndexes:
    def __init__(self, name: str):
        self.name = name
        self.inverted = InvertedIndex()
        self.tfidf = TFIDFIndexer()
        self.bm25 = BM25Indexer()
        self.word2vec = Word2VecIndexer()
        self.bert = BERTIndexer()
        self.loaded_models: list[str] = []

    def load_from_disk(self, cache_dir: Path) -> None:
        if self.inverted.load(cache_dir / "inverted_index.pkl"):
            self.loaded_models.append("inverted")
        if self.tfidf.load(cache_dir / "tfidf.pkl"):
            self.loaded_models.append("tfidf")
        if self.bm25.load(cache_dir / "bm25.pkl"):
            self.loaded_models.append("bm25")
        if self.word2vec.load(cache_dir / "word2vec"):
            self.loaded_models.append("word2vec")
        if self.bert.load(cache_dir / "bert"):
            self.loaded_models.append("bert")
        logger.info(f"[{self.name}] Loaded models: {self.loaded_models}")


_indexes: dict[str, _DatasetIndexes] = {}
_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Retrieval Service starting — loading indexes from disk…")
    for dataset_dir in DATA_DIR.iterdir():
        if dataset_dir.is_dir():
            idx = _DatasetIndexes(dataset_dir.name)
            idx.load_from_disk(dataset_dir)
            with _lock:
                _indexes[dataset_dir.name] = idx
    logger.info(f"Retrieval Service ready. Datasets: {list(_indexes.keys())}")
    yield
    logger.info("Retrieval Service shutting down.")


app = FastAPI(
    title="Retrieval Service",
    description="Dispatches queries to TF-IDF, BM25, embedding, or hybrid retrievers.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ===========================================================================
# Endpoints
# ===========================================================================
@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "service": "retrieval"}


@app.get("/retrieve/models", tags=["retrieval"])
def list_models():
    return {
        "models": [
            "tfidf", "bm25", "word2vec", "bert",
            "hybrid_serial", "hybrid_parallel"
        ]
    }


@app.get("/retrieve/datasets", tags=["retrieval"])
def list_datasets():
    with _lock:
        return {
            "datasets": [
                {"name": name, "loaded_models": idx.loaded_models}
                for name, idx in _indexes.items()
            ]
        }


@app.post("/retrieve", response_model=RetrievalResponse, tags=["retrieval"])
def retrieve(request: RetrievalRequest):
    """
    Retrieve top-K documents for the given query using the selected model.

    The query must already be preprocessed (tokens + cleaned string).
    For BERT, the original raw query is used for encoding.
    """
    with _lock:
        idx = _indexes.get(request.dataset)
    if idx is None:
        raise HTTPException(
            404,
            f"No indexes loaded for dataset '{request.dataset}'. "
            "Build indexes first via the Indexing Service.",
        )

    t0 = time.perf_counter()
    results_raw: list[tuple[str, float]] = _dispatch(request, idx)
    latency_ms = (time.perf_counter() - t0) * 1000

    results = [
        RetrievalResult(doc_id=doc_id, score=score, rank=rank)
        for rank, (doc_id, score) in enumerate(results_raw[:request.top_k], start=1)
    ]

    return RetrievalResponse(
        query_raw=request.query_raw,
        query_cleaned=request.query_cleaned,
        dataset=request.dataset,
        model=request.model,
        results=results,
        latency_ms=round(latency_ms, 2),
    )


def _dispatch(
    req: RetrievalRequest,
    idx: _DatasetIndexes,
) -> list[tuple[str, float]]:
    """Route the request to the appropriate retriever."""
    model = req.model

    if model == "tfidf":
        if "tfidf" not in idx.loaded_models:
            raise HTTPException(404, "TF-IDF index not available for this dataset.")
        return idx.tfidf.search(req.query_cleaned, top_k=req.top_k)

    elif model == "bm25":
        if "bm25" not in idx.loaded_models:
            raise HTTPException(404, "BM25 index not available.")
        return idx.bm25.search(
            req.query_tokens, top_k=req.top_k, k1=req.bm25_k1, b=req.bm25_b
        )

    elif model == "word2vec":
        if "word2vec" not in idx.loaded_models:
            raise HTTPException(404, "Word2Vec index not available.")
        return idx.word2vec.search(req.query_tokens, top_k=req.top_k)

    elif model == "bert":
        if "bert" not in idx.loaded_models:
            raise HTTPException(404, "BERT index not available.")
        return idx.bert.search(req.query_raw or req.query_cleaned, top_k=req.top_k)

    elif model == "hybrid_serial":
        if "bm25" not in idx.loaded_models:
            raise HTTPException(404, "BM25 index required for hybrid serial.")
        return serial_hybrid(
            query_tokens=req.query_tokens,
            query_raw=req.query_raw or req.query_cleaned,
            bm25_indexer=idx.bm25,
            bert_indexer=idx.bert,
            top_k=req.top_k,
            candidate_k=req.serial_candidate_k,
            bm25_k1=req.bm25_k1,
            bm25_b=req.bm25_b,
        )

    elif model == "hybrid_parallel":
        return parallel_hybrid(
            query_tokens=req.query_tokens,
            query_cleaned=req.query_cleaned,
            query_raw=req.query_raw or req.query_cleaned,
            tfidf_indexer=idx.tfidf,
            bm25_indexer=idx.bm25,
            bert_indexer=idx.bert,
            top_k=req.top_k,
            fusion_method=req.fusion_method,
            weights=req.hybrid_weights,
            bm25_k1=req.bm25_k1,
            bm25_b=req.bm25_b,
        )

    else:
        raise HTTPException(400, f"Unknown model: {model}")
