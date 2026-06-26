"""
=============================================================================
 Retrieval Service — hybrid_retriever.py
=============================================================================
 Implements two hybrid retrieval strategies:

 1. SERIAL (Pipeline):
    BM25 retrieves top-N candidates → BERT re-ranks the candidates.
    This is the standard "retrieve-then-rerank" pattern.

 2. PARALLEL (Late Fusion):
    TF-IDF, BM25, and BERT/Word2Vec run simultaneously on the full index.
    Their ranked lists are merged using one of three fusion methods:
      a) RRF  — Reciprocal Rank Fusion (robust, no parameter tuning)
      b) Linear — Weighted linear combination of normalized scores
      c) CombMNZ — CombSUM * number of systems that retrieved each doc
=============================================================================
"""

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger("retrieval_service.hybrid")


# ---------------------------------------------------------------------------
# Score normalization helpers
# ---------------------------------------------------------------------------
def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize a dict of {doc_id: score} to [0, 1]."""
    if not scores:
        return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    denom = max_s - min_s
    if denom == 0:
        return {k: 0.0 for k in scores}
    return {k: (v - min_s) / denom for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Fusion methods
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (Cormack et al., 2009).

    RRF score for doc d = Σ  1 / (k + rank(d, list_i))

    Parameters
    ----------
    ranked_lists : list of [(doc_id, score), ...]
        Each inner list is one retriever's ranked output.
    k : int
        Smoothing constant (default 60 is well-established in literature).

    Returns
    -------
    Merged and sorted list of (doc_id, rrf_score).
    """
    doc_scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


def linear_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float],
) -> list[tuple[str, float]]:
    """
    Weighted linear combination of min-max normalised scores.

    Parameters
    ----------
    ranked_lists : list of [(doc_id, score), ...]
    weights : list of float (must sum to 1.0, one per ranked list)

    Returns
    -------
    Merged and sorted list of (doc_id, fused_score).
    """
    assert len(ranked_lists) == len(weights), "Must have one weight per ranked list."
    doc_scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        normalized = _min_max_normalize(dict(ranked))
        for doc_id, score in normalized.items():
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + weight * score
    return sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)


def combmnz_fusion(
    ranked_lists: list[list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """
    CombMNZ: sum of normalised scores × number of systems that retrieved the doc.

    This rewards documents that multiple systems agree upon.
    """
    doc_scores: dict[str, float] = {}
    doc_hit_count: dict[str, int] = {}
    for ranked in ranked_lists:
        normalized = _min_max_normalize(dict(ranked))
        for doc_id, score in normalized.items():
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
            doc_hit_count[doc_id] = doc_hit_count.get(doc_id, 0) + 1
    combined = {d: doc_scores[d] * doc_hit_count[d] for d in doc_scores}
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# High-level hybrid retrieval functions
# ---------------------------------------------------------------------------
def serial_hybrid(
    query_tokens: list[str],
    query_raw: str,
    bm25_indexer,
    bert_indexer,
    top_k: int = 10,
    candidate_k: int = 100,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> list[tuple[str, float]]:
    """
    Serial Hybrid: BM25 retrieves candidates → BERT re-ranks.

    Steps:
    1. BM25 retrieves top `candidate_k` documents.
    2. BERT encodes the query and all candidate documents' stored vectors.
    3. Re-rank candidates by BERT cosine similarity.
    4. Return top `top_k`.
    """
    # Step 1: BM25 candidate retrieval
    candidates = bm25_indexer.search(
        query_tokens, top_k=candidate_k, k1=bm25_k1, b=bm25_b
    )
    candidate_ids = {doc_id for doc_id, _ in candidates}

    if not candidate_ids or bert_indexer._matrix is None:
        return candidates[:top_k]

    # Step 2: BERT query vector
    q_vec = bert_indexer.query_vector(query_raw)

    # Step 3: Score only candidates
    # Find indices of candidate docs in BERT's doc_id list
    id_to_idx = {did: i for i, did in enumerate(bert_indexer.doc_ids)}
    reranked: list[tuple[str, float]] = []
    for doc_id in candidate_ids:
        idx = id_to_idx.get(doc_id)
        if idx is not None:
            doc_vec = bert_indexer._matrix[idx]
            score = float(np.dot(q_vec, doc_vec))
            reranked.append((doc_id, score))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]


def parallel_hybrid(
    query_tokens: list[str],
    query_cleaned: str,
    query_raw: str,
    tfidf_indexer,
    bm25_indexer,
    bert_indexer,
    top_k: int = 10,
    fusion_method: Literal["rrf", "linear", "combmnz"] = "rrf",
    weights: dict[str, float] | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
) -> list[tuple[str, float]]:
    """
    Parallel Hybrid: all three retrievers run simultaneously, scores are fused.

    Uses an extended candidate pool (top_k × 10) per retriever to ensure
    good coverage before fusion.
    """
    candidate_k = min(top_k * 10, 1000)

    # Run all three retrievers
    tfidf_results = tfidf_indexer.search(query_cleaned, top_k=candidate_k)
    bm25_results = bm25_indexer.search(
        query_tokens, top_k=candidate_k, k1=bm25_k1, b=bm25_b
    )
    bert_results = bert_indexer.search(query_raw, top_k=candidate_k) if bert_indexer._matrix is not None else []

    ranked_lists = [tfidf_results, bm25_results, bert_results]
    # Filter out empty lists
    valid_lists = [r for r in ranked_lists if r]

    if not valid_lists:
        return []

    if fusion_method == "rrf":
        fused = reciprocal_rank_fusion(valid_lists)
    elif fusion_method == "linear":
        w = weights or {"tfidf": 0.3, "bm25": 0.4, "bert": 0.3}
        # Match weights to the valid lists (tfidf, bm25, bert order)
        model_weights = [w.get("tfidf", 0.3), w.get("bm25", 0.4), w.get("bert", 0.3)]
        valid_weights = [model_weights[i] for i, r in enumerate(ranked_lists) if r]
        # Normalise weights
        total = sum(valid_weights)
        valid_weights = [wt / total for wt in valid_weights]
        fused = linear_fusion(valid_lists, valid_weights)
    elif fusion_method == "combmnz":
        fused = combmnz_fusion(valid_lists)
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")

    return fused[:top_k]
