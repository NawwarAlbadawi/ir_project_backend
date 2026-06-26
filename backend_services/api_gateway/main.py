"""
=============================================================================
 API Gateway — main.py
 Port: 8000
=============================================================================
 Single entry point for the UI.  Orchestrates all microservice calls.

 Key flows:
   /search  → preprocess → [refine] → retrieve → [enrich snippets] → return
   /search/evaluate → /search + evaluate/full
   /index/build → indexing service
   /dataset/load → preprocessing service
=============================================================================
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api_gateway.schemas import (
    SearchRequest, SearchResponse, SearchResultItem,
    LoadDatasetRequest, BuildIndexRequest, EvalCompareRequest,
)
import api_gateway.service_clients as svc

logger = logging.getLogger("api_gateway")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"

# ---------------------------------------------------------------------------
# Shared async HTTP client (one per app lifecycle)
# ---------------------------------------------------------------------------
_http_client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http_client
    _http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
    logger.info("API Gateway ready.")
    yield
    await _http_client.aclose()
    logger.info("API Gateway shut down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="IR System — API Gateway",
    description=(
        "Unified entry point for the Information Retrieval system. "
        "Orchestrates Preprocessing, Indexing, Retrieval, Evaluation, "
        "and Query Refinement microservices."
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

# Serve the frontend static files
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ===========================================================================
# Root — serve the UI
# ===========================================================================
@app.get("/", include_in_schema=False)
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "IR System API Gateway", "docs": "/docs"}


# ===========================================================================
# Health
# ===========================================================================
@app.get("/health", tags=["meta"])
async def health():
    service_health = await svc.check_all_health(_http_client)
    overall = "ok" if all(v == "ok" for v in service_health.values()) else "degraded"
    return {"status": overall, "services": service_health}


# ===========================================================================
# Datasets
# ===========================================================================
@app.get("/datasets", tags=["datasets"])
async def list_datasets():
    return {
        "datasets": [
            {
                "name": "quora",
                "description": "BEIR/Quora duplicate question retrieval (~530K docs)",
                "ir_key": "beir/quora/test",
            },
            {
                "name": "msmarco",
                "description": "MS MARCO passage retrieval (200K passage subset)",
                "ir_key": "msmarco-passage/train/triples-small",
            },
        ]
    }


@app.post("/dataset/load", tags=["datasets"])
async def load_dataset(request: LoadDatasetRequest):
    """Trigger dataset loading in the Preprocessing Service."""
    try:
        result = await svc.load_dataset(_http_client, request.dataset_name)
        return result
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


@app.get("/dataset/status", tags=["datasets"])
async def dataset_status(dataset_name: str):
    try:
        return await svc.dataset_status(_http_client, dataset_name)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


# ===========================================================================
# Index management
# ===========================================================================
@app.post("/index/build", tags=["index"])
async def build_index(request: BuildIndexRequest):
    """Trigger index build in the Indexing Service."""
    try:
        return await svc.build_index(_http_client, request.dataset_name, request.models)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


@app.get("/index/status", tags=["index"])
async def index_status(dataset_name: str):
    try:
        return await svc.index_status(_http_client, dataset_name)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


@app.get("/index/stats", tags=["index"])
async def index_stats(dataset_name: str):
    try:
        return await svc.index_stats(_http_client, dataset_name)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)


# ===========================================================================
# Models
# ===========================================================================
@app.get("/models", tags=["meta"])
async def list_models():
    return {
        "models": [
            {"id": "tfidf",           "name": "VSM / TF-IDF",        "category": "sparse"},
            {"id": "bm25",            "name": "BM25",                 "category": "sparse",  "tunable": True},
            {"id": "word2vec",        "name": "Word2Vec Embeddings",  "category": "dense"},
            {"id": "bert",            "name": "BERT (MiniLM-L6-v2)", "category": "dense"},
            {"id": "hybrid_serial",   "name": "Hybrid Serial",        "category": "hybrid",  "description": "BM25 → BERT re-rank"},
            {"id": "hybrid_parallel", "name": "Hybrid Parallel",      "category": "hybrid",  "description": "RRF / Linear / CombMNZ fusion"},
        ]
    }


# ===========================================================================
# MAIN: Search
# ===========================================================================
@app.post("/search", response_model=SearchResponse, tags=["search"])
async def search(request: SearchRequest):
    """
    Full search pipeline:

    1. Preprocess the query (Preprocessing Service)
    2. [Optional] Refine the query (Query Refinement Service)
    3. Retrieve top-K documents (Retrieval Service)
    4. Fetch text snippets for the top results (Preprocessing Service)
    5. Return unified response

    Toggle ``use_refinement`` to compare base vs. refined execution.
    """
    # ----- Step 1: Preprocess query -----
    preprocess_opts = {
        "lowercase": True,
        "remove_punctuation": True,
        "remove_stopwords": True,
        "stem": request.preprocess_stem,
        "lemmatize": request.preprocess_lemmatize,
    }
    try:
        pp_result = await svc.preprocess_text(
            _http_client, request.query, preprocess_opts
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Preprocessing Service error: {e.response.text}")

    query_tokens: list[str] = pp_result["tokens"]
    query_cleaned: str = pp_result["cleaned"]
    refinement_info: dict | None = None

    # ----- Step 2: [Optional] Refine query -----
    query_for_retrieval = query_cleaned
    query_raw_for_bert = request.query
    refined_query_str: str | None = None

    if request.use_refinement:
        try:
            ref_result = await svc.refine_query(
                _http_client,
                query=request.query,
                techniques=request.refinement_techniques,
                session_id=request.session_id,
            )
            refined_query_str = ref_result["refined_query"]
            refinement_info = {
                "corrections": ref_result.get("corrections", {}),
                "expansions": ref_result.get("expansions", {}),
                "history_boosts": ref_result.get("history_boosts", []),
                "techniques_applied": ref_result.get("techniques_applied", []),
            }
            # Re-preprocess the refined query to get clean tokens
            pp_refined = await svc.preprocess_text(
                _http_client, refined_query_str, preprocess_opts
            )
            query_tokens = pp_refined["tokens"]
            query_for_retrieval = pp_refined["cleaned"]
            query_raw_for_bert = refined_query_str
        except httpx.HTTPStatusError as e:
            logger.warning(f"Refinement failed ({e.response.status_code}), using raw query.")

    # ----- Step 3: Retrieve -----
    retrieval_payload = {
        "query_tokens": query_tokens,
        "query_cleaned": query_for_retrieval,
        "query_raw": query_raw_for_bert,
        "dataset": request.dataset,
        "model": request.model,
        "top_k": request.top_k,
        "bm25_k1": request.bm25_k1,
        "bm25_b": request.bm25_b,
        "fusion_method": request.fusion_method,
        "hybrid_weights": request.hybrid_weights,
        "serial_candidate_k": request.serial_candidate_k,
    }
    try:
        ret_result = await svc.retrieve(_http_client, retrieval_payload)
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Retrieval Service error: {e.response.text}")

    # ----- Step 4: Fetch snippets for display -----
    # Get a batch of raw docs to find snippets for the result set
    result_doc_ids = {r["doc_id"] for r in ret_result["results"]}
    snippets: dict[str, str] = {}
    try:
        raw_docs_resp = await svc.get_raw_docs(
            _http_client, request.dataset, offset=0, limit=200
        )
        for doc in raw_docs_resp.get("documents", []):
            if doc["doc_id"] in result_doc_ids:
                snippets[doc["doc_id"]] = doc["text"][:300]  # first 300 chars
    except Exception:
        pass  # Snippets are optional; don't fail the search

    # ----- Step 5: Assemble response -----
    results = [
        SearchResultItem(
            rank=r["rank"],
            doc_id=r["doc_id"],
            score=r["score"],
            snippet=snippets.get(r["doc_id"]),
        )
        for r in ret_result["results"]
    ]

    # Record in history if refinement is on
    if request.use_refinement:
        try:
            await svc.add_history(
                _http_client,
                session_id=request.session_id,
                query=request.query,
                clicked_doc_ids=[],
            )
        except Exception:
            pass

    return SearchResponse(
        query_original=request.query,
        query_refined=refined_query_str,
        query_cleaned=query_for_retrieval,
        dataset=request.dataset,
        model=request.model,
        results=results,
        latency_ms=ret_result.get("latency_ms", 0.0),
        refinement_info=refinement_info,
    )


# ===========================================================================
# Evaluation
# ===========================================================================
@app.post("/search/evaluate", tags=["evaluation"])
async def search_and_evaluate(request: EvalCompareRequest):
    """
    Benchmark a model against the full qrels.

    Runs retrieval for the first ``num_queries`` queries from the dataset
    and evaluates with MAP, Recall, P@10, nDCG@10.

    If ``compare_refinement=True``, runs the evaluation twice:
    once without refinement and once with spelling+synonym refinement,
    then returns a side-by-side comparison.
    """
    # Fetch qrels from preprocessing service
    try:
        qrels_resp = await svc.get_qrels(_http_client, request.dataset)
        qrels: dict = qrels_resp["qrels"]
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Could not fetch qrels: {e.response.text}")

    if not qrels:
        raise HTTPException(404, f"No qrels found for dataset '{request.dataset}'.")

    # Limit to num_queries
    query_ids = list(qrels.keys())[: request.num_queries]
    limited_qrels = {qid: qrels[qid] for qid in query_ids}

    # Fetch query texts from preprocessing service
    query_text_map = {}
    try:
        q_resp = await _http_client.get(
            f"{svc.PREPROCESSING_URL}/dataset/queries",
            params={"dataset_name": request.dataset, "limit": 10000}
        )
        if q_resp.status_code == 200:
            for q_obj in q_resp.json().get("queries", []):
                query_text_map[str(q_obj["query_id"])] = q_obj.get("original", "")
    except Exception as e:
        logger.error(f"Failed to fetch query texts: {e}")

    async def _run_retrieval_for_queries(use_refinement: bool) -> dict[str, list[dict]]:
        """Run retrieval for all query_ids."""
        ranked: dict[str, list[dict]] = {}
        for qid in query_ids:
            real_query = query_text_map.get(str(qid), qid)
            search_req = SearchRequest(
                query=real_query,
                dataset=request.dataset,
                model=request.model,
                top_k=request.k,
                use_refinement=use_refinement,
                refinement_techniques=["spell", "synonyms"],
            )
            try:
                resp = await search(search_req)
                ranked[qid] = [
                    {"doc_id": r.doc_id, "score": r.score, "rank": r.rank}
                    for r in resp.results
                ]
            except Exception:
                ranked[qid] = []
        return ranked

    # Base evaluation (no refinement)
    ranked_base = await _run_retrieval_for_queries(use_refinement=False)
    eval_payload_base = {
        "dataset": request.dataset,
        "model": request.model,
        "ranked_results": ranked_base,
        "qrels": limited_qrels,
        "k": request.k,
    }
    try:
        base_metrics = await svc.evaluate_full(_http_client, eval_payload_base)
    except httpx.HTTPStatusError as e:
        raise HTTPException(502, f"Evaluation service error: {e.response.text}")

    response = {"base": base_metrics}

    if request.compare_refinement:
        ranked_refined = await _run_retrieval_for_queries(use_refinement=True)
        eval_payload_refined = {**eval_payload_base, "ranked_results": ranked_refined}
        try:
            refined_metrics = await svc.evaluate_full(_http_client, eval_payload_refined)
            response["refined"] = refined_metrics
        except httpx.HTTPStatusError as e:
            logger.warning(f"Refined evaluation failed: {e}")

    return response


# ===========================================================================
# Topic Detection (LDA)
# ===========================================================================
@app.post("/topics/build", tags=["topics"])
async def build_topics(dataset_name: str, num_topics: int = 10):
    """
    Build an LDA topic model for a dataset.
    Runs in background — poll GET /topics/status to check progress.
    """
    try:
        return await svc.build_topics(_http_client, dataset_name, num_topics=num_topics)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"Topic Service error: {e.response.text}")


@app.get("/topics/status", tags=["topics"])
async def get_topic_status(dataset_name: str):
    """Check whether the LDA topic model is built for a dataset."""
    try:
        return await svc.topic_status(_http_client, dataset_name)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"Topic Service error: {e.response.text}")


@app.get("/topics/all", tags=["topics"])
async def get_all_topics(dataset_name: str):
    """Get all discovered topics with their top words and document counts."""
    try:
        return await svc.get_all_topics(_http_client, dataset_name)
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, f"Topic Service error: {e.response.text}")


@app.post("/search/with-topics", tags=["topics"])
async def search_with_topics(request: SearchRequest):
    """
    Run a full search AND attach a topic label to each result.

    Returns the same response as /search but with an extra
    ``topic`` field on every result item containing:
    - topic_id
    - topic_label  (auto-generated from top words)
    - top_words    (list of most representative words)
    - probability  (confidence of the assignment)
    """
    # Run the normal search pipeline
    search_response: SearchResponse = await search(request)

    # Detect topics for all returned documents
    doc_ids = [r.doc_id for r in search_response.results]
    topic_map: dict[str, dict] = {}
    try:
        topic_resp = await svc.detect_doc_topics(_http_client, request.dataset, doc_ids)
        for item in topic_resp.get("results", []):
            topic_map[item["doc_id"]] = {
                "topic_id":    item["topic_id"],
                "topic_label": item["topic_label"],
                "top_words":   item["top_words"],
                "probability": item["probability"],
            }
    except Exception as e:
        logger.warning(f"Topic detection failed (search still returns): {e}")

    # Enrich results with topic info
    enriched_results = []
    for r in search_response.results:
        item_dict = r.model_dump()
        item_dict["topic"] = topic_map.get(r.doc_id)
        enriched_results.append(item_dict)

    return {
        "query_original":  search_response.query_original,
        "query_refined":   search_response.query_refined,
        "query_cleaned":   search_response.query_cleaned,
        "dataset":         search_response.dataset,
        "model":           search_response.model,
        "latency_ms":      search_response.latency_ms,
        "refinement_info": search_response.refinement_info,
        "results":         enriched_results,
    }
