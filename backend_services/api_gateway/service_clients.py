"""
=============================================================================
 API Gateway — service_clients.py
=============================================================================
 Typed async HTTP clients for each microservice.
 Uses httpx.AsyncClient with timeouts and retry logic.
=============================================================================
"""

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger("api_gateway.clients")

# Service base URLs — can be overridden via env vars in production
PREPROCESSING_URL = "http://localhost:8001"
INDEXING_URL      = "http://localhost:8002"
RETRIEVAL_URL     = "http://localhost:8003"
EVAL_URL          = "http://localhost:8004"
REFINEMENT_URL    = "http://localhost:8005"
TOPIC_URL         = "http://localhost:8006"

# Generous timeouts for long-running operations (index build, evaluation)
DEFAULT_TIMEOUT   = httpx.Timeout(30.0, connect=5.0)
LONG_TIMEOUT      = httpx.Timeout(300.0, connect=5.0)


async def _get(client: httpx.AsyncClient, url: str, **kwargs) -> dict:
    resp = await client.get(url, **kwargs)
    resp.raise_for_status()
    return resp.json()


async def _post(client: httpx.AsyncClient, url: str, payload: dict, **kwargs) -> dict:
    resp = await client.post(url, json=payload, **kwargs)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Preprocessing Service
# ---------------------------------------------------------------------------
async def preprocess_text(client: httpx.AsyncClient, text: str, options: dict) -> dict:
    return await _post(
        client,
        f"{PREPROCESSING_URL}/preprocess/text",
        {"text": text, "options": options},
    )


async def load_dataset(client: httpx.AsyncClient, dataset_name: str) -> dict:
    resp = await client.post(
        f"{PREPROCESSING_URL}/dataset/load",
        params={"dataset_name": dataset_name},
        timeout=LONG_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


async def dataset_status(client: httpx.AsyncClient, dataset_name: str) -> dict:
    return await _get(
        client,
        f"{PREPROCESSING_URL}/dataset/status",
        params={"dataset_name": dataset_name},
    )


async def get_qrels(client: httpx.AsyncClient, dataset_name: str) -> dict:
    return await _get(
        client,
        f"{PREPROCESSING_URL}/dataset/qrels",
        params={"dataset_name": dataset_name},
        timeout=LONG_TIMEOUT,
    )


async def get_raw_docs(
    client: httpx.AsyncClient, dataset_name: str, offset: int, limit: int
) -> dict:
    return await _get(
        client,
        f"{PREPROCESSING_URL}/dataset/raw_docs",
        params={"dataset_name": dataset_name, "offset": offset, "limit": limit},
    )


# ---------------------------------------------------------------------------
# Indexing Service
# ---------------------------------------------------------------------------
async def build_index(
    client: httpx.AsyncClient, dataset_name: str, models: list[str]
) -> dict:
    return await _post(
        client,
        f"{INDEXING_URL}/index/build",
        {"dataset_name": dataset_name, "models": models},
        timeout=LONG_TIMEOUT,
    )


async def index_status(client: httpx.AsyncClient, dataset_name: str) -> dict:
    return await _get(
        client,
        f"{INDEXING_URL}/index/status",
        params={"dataset_name": dataset_name},
    )


async def index_stats(client: httpx.AsyncClient, dataset_name: str) -> dict:
    return await _get(
        client,
        f"{INDEXING_URL}/index/stats",
        params={"dataset_name": dataset_name},
    )


# ---------------------------------------------------------------------------
# Query Refinement Service
# ---------------------------------------------------------------------------
async def refine_query(
    client: httpx.AsyncClient,
    query: str,
    techniques: list[str],
    session_id: str = "default",
    max_expansions: int = 3,
) -> dict:
    return await _post(
        client,
        f"{REFINEMENT_URL}/refine",
        {
            "query": query,
            "techniques": techniques,
            "session_id": session_id,
            "max_expansions": max_expansions,
        },
    )


async def add_history(
    client: httpx.AsyncClient,
    session_id: str,
    query: str,
    clicked_doc_ids: list[str],
) -> dict:
    return await _post(
        client,
        f"{REFINEMENT_URL}/refine/history",
        {"session_id": session_id, "query": query, "clicked_doc_ids": clicked_doc_ids},
    )


# ---------------------------------------------------------------------------
# Retrieval Service
# ---------------------------------------------------------------------------
async def retrieve(client: httpx.AsyncClient, payload: dict) -> dict:
    return await _post(
        client, f"{RETRIEVAL_URL}/retrieve", payload, timeout=LONG_TIMEOUT
    )


# ---------------------------------------------------------------------------
# Evaluation Service
# ---------------------------------------------------------------------------
async def evaluate_full(client: httpx.AsyncClient, payload: dict) -> dict:
    return await _post(
        client, f"{EVAL_URL}/evaluate/full", payload, timeout=LONG_TIMEOUT
    )


# ---------------------------------------------------------------------------
# Topic Detection Service
# ---------------------------------------------------------------------------
async def build_topics(
    client: httpx.AsyncClient,
    dataset_name: str,
    num_topics: int = 10,
    num_top_words: int = 8,
) -> dict:
    return await _post(
        client,
        f"{TOPIC_URL}/topics/build",
        {"dataset_name": dataset_name, "num_topics": num_topics, "num_top_words": num_top_words},
        timeout=LONG_TIMEOUT,
    )


async def topic_status(client: httpx.AsyncClient, dataset_name: str) -> dict:
    return await _get(
        client,
        f"{TOPIC_URL}/topics/status",
        params={"dataset_name": dataset_name},
    )


async def get_all_topics(client: httpx.AsyncClient, dataset_name: str) -> dict:
    return await _get(
        client,
        f"{TOPIC_URL}/topics/all",
        params={"dataset_name": dataset_name},
    )


async def detect_doc_topics(
    client: httpx.AsyncClient, dataset_name: str, doc_ids: list[str]
) -> dict:
    return await _post(
        client,
        f"{TOPIC_URL}/topics/detect",
        {"dataset_name": dataset_name, "doc_ids": doc_ids},
    )


async def infer_query_topic(
    client: httpx.AsyncClient, dataset_name: str, text: str
) -> dict:
    return await _get(
        client,
        f"{TOPIC_URL}/topics/infer",
        params={"dataset_name": dataset_name, "text": text},
    )


# ---------------------------------------------------------------------------
# Health aggregation
# ---------------------------------------------------------------------------
async def check_all_health(client: httpx.AsyncClient) -> dict:
    services = {
        "preprocessing":    f"{PREPROCESSING_URL}/health",
        "indexing":         f"{INDEXING_URL}/health",
        "retrieval":        f"{RETRIEVAL_URL}/health",
        "evaluation":       f"{EVAL_URL}/health",
        "refinement":       f"{REFINEMENT_URL}/health",
        "topic_detection":  f"{TOPIC_URL}/health",
    }
    results = {}
    for name, url in services.items():
        try:
            resp = await client.get(url, timeout=3.0)
            results[name] = "ok" if resp.status_code == 200 else "error"
        except Exception:
            results[name] = "unreachable"
    return results
