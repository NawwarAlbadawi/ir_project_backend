"""
=============================================================================
 Indexing Service — main.py
 Port: 8002
=============================================================================
 Coordinates building and exposing all four indexes:
   - Inverted Index
   - TF-IDF (scikit-learn)
   - BM25 (rank-bm25)
   - Word2Vec + BERT embeddings

 Indexes are built from preprocessed documents fetched from the
 Preprocessing Service and persisted to /data/<dataset>/.
=============================================================================
"""

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware

from indexing_service.schemas import (
    BuildIndexRequest,
    IndexStatusResponse,
    InvertedIndexQueryResponse,
    IndexStatsResponse,
)
from indexing_service.inverted_index import InvertedIndex
from indexing_service.tfidf_indexer import TFIDFIndexer
from indexing_service.bm25_indexer import BM25Indexer
from indexing_service.embedding_indexer import Word2VecIndexer, BERTIndexer

logger = logging.getLogger("indexing_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PREPROCESSING_SERVICE_URL = "http://localhost:8001"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PAGE_SIZE = 5_000  # how many docs to fetch from preprocessing service per page


# ---------------------------------------------------------------------------
# Per-dataset state
# ---------------------------------------------------------------------------
class _DatasetIndexState:
    def __init__(self, name: str):
        self.name = name
        self.status: str = "not_built"    # not_built | building | ready | error
        self.built_models: list[str] = []
        self.progress: int = 0
        self.total: int = 0
        self.error: str | None = None

        # Index objects
        self.inverted = InvertedIndex()
        self.tfidf = TFIDFIndexer()
        self.bm25 = BM25Indexer()
        self.word2vec = Word2VecIndexer()
        self.bert = BERTIndexer()

        self._lock = threading.Lock()

    def to_response(self) -> IndexStatusResponse:
        with self._lock:
            return IndexStatusResponse(
                dataset=self.name,
                status=self.status,
                built_models=list(self.built_models),
                progress=self.progress,
                total=self.total,
                error=self.error,
            )


_states: dict[str, _DatasetIndexState] = {}
_states_lock = threading.Lock()


def _get_or_create(dataset_name: str) -> _DatasetIndexState:
    with _states_lock:
        if dataset_name not in _states:
            _states[dataset_name] = _DatasetIndexState(dataset_name)
        return _states[dataset_name]


# ---------------------------------------------------------------------------
# Lifespan — try to restore cached indexes on startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Indexing Service starting — scanning for cached indexes…")
    for dataset_dir in DATA_DIR.iterdir():
        if dataset_dir.is_dir():
            state = _get_or_create(dataset_dir.name)
            _try_restore_from_disk(state, dataset_dir)
    logger.info("Indexing Service ready.")
    yield
    logger.info("Indexing Service shutting down.")


def _try_restore_from_disk(state: _DatasetIndexState, cache_dir: Path) -> None:
    """Attempt to load all persisted indexes from disk."""
    restored = []
    if state.inverted.load(cache_dir / "inverted_index.pkl"):
        restored.append("inverted")
    if state.tfidf.load(cache_dir / "tfidf.pkl"):
        restored.append("tfidf")
    if state.bm25.load(cache_dir / "bm25.pkl"):
        restored.append("bm25")
    if state.word2vec.load(cache_dir / "word2vec"):
        restored.append("word2vec")
    if state.bert.load(cache_dir / "bert"):
        restored.append("bert")
    if restored:
        with state._lock:
            state.status = "ready"
            state.built_models = restored
        logger.info(f"[{state.name}] Restored from disk: {restored}")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Indexing Service",
    description="Builds and exposes Inverted Index, TF-IDF, BM25, and Embedding indexes.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ===========================================================================
# Health
# ===========================================================================
@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "service": "indexing"}


# ===========================================================================
# Index management
# ===========================================================================
@app.post("/index/build", tags=["index"])
def build_index(request: BuildIndexRequest, background_tasks: BackgroundTasks):
    """
    Trigger building indexes for a dataset.

    Fetches preprocessed documents from the Preprocessing Service,
    then builds the requested model indexes.  Runs in the background.
    """
    state = _get_or_create(request.dataset_name)
    if state.status == "building":
        raise HTTPException(409, f"Already building index for '{request.dataset_name}'.")
    background_tasks.add_task(_build_pipeline, state, request)
    return {"message": f"Index build started for '{request.dataset_name}'."}


@app.get("/index/status", response_model=IndexStatusResponse, tags=["index"])
def index_status(dataset_name: str = Query(...)):
    state = _get_or_create(dataset_name)
    return state.to_response()


@app.get("/index/stats", response_model=IndexStatsResponse, tags=["index"])
def index_stats(dataset_name: str = Query(...)):
    state = _get_or_create(dataset_name)
    if state.status != "ready":
        raise HTTPException(404, f"Index for '{dataset_name}' not ready.")
    return IndexStatsResponse(
        dataset=dataset_name,
        num_docs=state.inverted.N,
        vocab_size=state.inverted.vocab_size,
        available_models=state.built_models,
    )


@app.get("/index/inverted", response_model=InvertedIndexQueryResponse, tags=["index"])
def query_inverted_index(
    dataset_name: str = Query(...),
    term: str = Query(..., description="The (preprocessed) term to look up."),
):
    """Inspect the raw inverted index entry for a specific term."""
    state = _get_or_create(dataset_name)
    if "inverted" not in state.built_models:
        raise HTTPException(404, "Inverted index not built for this dataset.")
    info = state.inverted.term_info(term)
    if info is None:
        raise HTTPException(404, f"Term '{term}' not in index.")
    return InvertedIndexQueryResponse(
        term=term, postings=info["postings"], df=info["df"]
    )


# ===========================================================================
# Build pipeline (background)
# ===========================================================================
def _build_pipeline(state: _DatasetIndexState, request: BuildIndexRequest) -> None:
    """Fetch documents from preprocessing service and build all indexes."""
    with state._lock:
        state.status = "building"
        state.error = None
        state.built_models = []

    try:
        docs = _fetch_all_documents(state, request.dataset_name)
        cache_dir = DATA_DIR / request.dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)

        models_to_build = request.models
        total_steps = len(models_to_build)
        step = 0

        # ---- Inverted Index ----
        if "inverted" in models_to_build or "tfidf" in models_to_build or "bm25" in models_to_build:
            logger.info(f"[{state.name}] Building Inverted Index…")
            state.inverted.build(docs)
            state.inverted.save(cache_dir / "inverted_index.pkl")
            with state._lock:
                state.built_models.append("inverted")
            step += 1
            _update_progress(state, step, total_steps)

        # ---- TF-IDF ----
        if "tfidf" in models_to_build:
            logger.info(f"[{state.name}] Building TF-IDF…")
            state.tfidf.build(docs)
            state.tfidf.save(cache_dir / "tfidf.pkl")
            with state._lock:
                state.built_models.append("tfidf")
            step += 1
            _update_progress(state, step, total_steps)

        # ---- BM25 ----
        if "bm25" in models_to_build:
            logger.info(f"[{state.name}] Building BM25…")
            state.bm25.build(docs)
            state.bm25.save(cache_dir / "bm25.pkl")
            with state._lock:
                state.built_models.append("bm25")
            step += 1
            _update_progress(state, step, total_steps)

        # ---- Word2Vec ----
        if "word2vec" in models_to_build:
            logger.info(f"[{state.name}] Building Word2Vec…")
            state.word2vec.build(docs)
            state.word2vec.save(cache_dir / "word2vec")
            with state._lock:
                state.built_models.append("word2vec")
            step += 1
            _update_progress(state, step, total_steps)

        # ---- BERT ----
        if "bert" in models_to_build:
            logger.info(f"[{state.name}] Building BERT…")
            # Inject raw_text for BERT (BERT works better on un-stemmed text)
            state.bert.build(docs)
            state.bert.save(cache_dir / "bert")
            with state._lock:
                state.built_models.append("bert")
            step += 1
            _update_progress(state, step, total_steps)

        with state._lock:
            state.status = "ready"
        logger.info(f"[{state.name}] All indexes built successfully.")

    except Exception as exc:
        logger.exception(f"[{state.name}] Index build failed: {exc}")
        with state._lock:
            state.status = "error"
            state.error = str(exc)


def _fetch_all_documents(state: _DatasetIndexState, dataset_name: str) -> list[dict]:
    """
    Page through the Preprocessing Service to collect all preprocessed docs.
    """
    docs: list[dict] = []
    offset = 0
    logger.info(f"[{dataset_name}] Fetching documents from Preprocessing Service…")

    with httpx.Client(timeout=120.0) as client:
        # First get total doc count via status
        status_resp = client.get(
            f"{PREPROCESSING_SERVICE_URL}/dataset/status",
            params={"dataset_name": dataset_name},
        )
        status_resp.raise_for_status()
        total = status_resp.json().get("total_docs", 0)

        with state._lock:
            state.total = total

        while True:
            resp = client.get(
                f"{PREPROCESSING_SERVICE_URL}/dataset/docs",
                params={"dataset_name": dataset_name, "offset": offset, "limit": PAGE_SIZE},
            )
            resp.raise_for_status()
            batch = resp.json()["documents"]
            if not batch:
                break
            docs.extend(batch)
            offset += len(batch)
            with state._lock:
                state.progress = len(docs)
            logger.debug(f"[{dataset_name}] Fetched {len(docs)}/{total} docs…")

    logger.info(f"[{dataset_name}] Fetched {len(docs)} documents total.")
    return docs


def _update_progress(state: _DatasetIndexState, step: int, total: int) -> None:
    with state._lock:
        state.progress = step
        state.total = total
