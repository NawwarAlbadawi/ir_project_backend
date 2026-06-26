"""
=============================================================================
 Ranking & Evaluation Service — evaluator.py
=============================================================================
 Implements all four IR evaluation metrics from scratch
 (no external evaluation library required):

   1. Average Precision (AP) → aggregated as MAP
   2. Recall @ K
   3. Precision @ K
   4. nDCG @ K (Normalized Discounted Cumulative Gain)

 All metrics follow the standard TREC evaluation definitions.
=============================================================================
"""

import logging
import math
from typing import Optional

logger = logging.getLogger("ranking_eval_service.evaluator")


# ---------------------------------------------------------------------------
# Per-query metrics
# ---------------------------------------------------------------------------
def average_precision(
    ranked_doc_ids: list[str],
    relevant_docs: dict[str, int],
    relevance_threshold: int = 1,
) -> float:
    """
    Compute Average Precision for a single query.

    AP = (1 / R) × Σ P@k × rel(k)

    where R = total number of relevant documents in the collection
    and rel(k) is 1 if the document at rank k is relevant.

    Parameters
    ----------
    ranked_doc_ids : list[str]
        Ordered list of retrieved doc IDs (best first).
    relevant_docs : dict[str, int]
        {doc_id: relevance_score} — all judged relevant docs for this query.
    relevance_threshold : int
        Minimum relevance score to count as relevant (default 1).

    Returns
    -------
    float : AP value in [0, 1]
    """
    R = sum(1 for v in relevant_docs.values() if v >= relevance_threshold)
    if R == 0:
        return 0.0

    num_relevant_found = 0
    precision_sum = 0.0

    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if relevant_docs.get(doc_id, 0) >= relevance_threshold:
            num_relevant_found += 1
            precision_sum += num_relevant_found / rank

    return precision_sum / R


def recall_at_k(
    ranked_doc_ids: list[str],
    relevant_docs: dict[str, int],
    k: int,
    relevance_threshold: int = 1,
) -> float:
    """
    Recall at rank cutoff K.

    Recall@K = |{relevant docs in top K}| / |{all relevant docs}|
    """
    R = sum(1 for v in relevant_docs.values() if v >= relevance_threshold)
    if R == 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    retrieved_relevant = sum(
        1 for d in top_k if relevant_docs.get(d, 0) >= relevance_threshold
    )
    return retrieved_relevant / R


def precision_at_k(
    ranked_doc_ids: list[str],
    relevant_docs: dict[str, int],
    k: int,
    relevance_threshold: int = 1,
) -> float:
    """
    Precision at rank cutoff K.

    P@K = |{relevant docs in top K}| / K
    """
    top_k = ranked_doc_ids[:k]
    if not top_k:
        return 0.0
    relevant = sum(
        1 for d in top_k if relevant_docs.get(d, 0) >= relevance_threshold
    )
    return relevant / k


def dcg_at_k(
    ranked_doc_ids: list[str],
    relevant_docs: dict[str, int],
    k: int,
) -> float:
    """
    Discounted Cumulative Gain at K.

    DCG@K = Σ (2^rel_i − 1) / log2(i + 1)

    Uses graded relevance if the qrel values > 1.
    """
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = relevant_docs.get(doc_id, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def ideal_dcg_at_k(
    relevant_docs: dict[str, int],
    k: int,
) -> float:
    """
    Ideal DCG: compute DCG on the best possible ranking.

    Sort all relevant docs by their relevance score descending,
    then compute DCG as if they were ranked 1…|R|.
    """
    sorted_rels = sorted(relevant_docs.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(sorted_rels[:k], start=1):
        if rel > 0:
            idcg += (2 ** rel - 1) / math.log2(i + 1)
    return idcg


def ndcg_at_k(
    ranked_doc_ids: list[str],
    relevant_docs: dict[str, int],
    k: int,
) -> float:
    """
    Normalized DCG at K.

    nDCG@K = DCG@K / IDCG@K
    """
    idcg = ideal_dcg_at_k(relevant_docs, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_doc_ids, relevant_docs, k) / idcg


# ---------------------------------------------------------------------------
# Aggregate evaluation over all queries
# ---------------------------------------------------------------------------
def evaluate_all(
    ranked_results: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    k: int = 10,
    relevance_threshold: int = 1,
) -> dict:
    """
    Evaluate retrieval quality over all queries.

    Parameters
    ----------
    ranked_results : dict[str, list[str]]
        {query_id: [doc_id, ...]} — retrieved docs in ranked order.
    qrels : dict[str, dict[str, int]]
        {query_id: {doc_id: relevance_score}}
    k : int
        Rank cutoff for P@k, nDCG@k, Recall@k.
    relevance_threshold : int
        Minimum relevance score to count as relevant.

    Returns
    -------
    dict with aggregate and per-query metrics.
    """
    per_query = []
    ap_values, recall_values, p_values, ndcg_values = [], [], [], []

    for query_id, doc_ids in ranked_results.items():
        rel = qrels.get(query_id, {})
        if not rel:
            logger.debug(f"No qrels for query {query_id}, skipping.")
            continue

        ap = average_precision(doc_ids, rel, relevance_threshold)
        rec = recall_at_k(doc_ids, rel, k, relevance_threshold)
        prec = precision_at_k(doc_ids, rel, k, relevance_threshold)
        ndcg = ndcg_at_k(doc_ids, rel, k)

        ap_values.append(ap)
        recall_values.append(rec)
        p_values.append(prec)
        ndcg_values.append(ndcg)

        per_query.append(
            {
                "query_id": query_id,
                "average_precision": round(ap, 4),
                "recall": round(rec, 4),
                "precision_at_k": round(prec, 4),
                "ndcg_at_k": round(ndcg, 4),
            }
        )

    num_q = len(ap_values)
    return {
        "map": round(sum(ap_values) / max(num_q, 1), 4),
        "mean_recall": round(sum(recall_values) / max(num_q, 1), 4),
        "mean_precision_at_k": round(sum(p_values) / max(num_q, 1), 4),
        "mean_ndcg_at_k": round(sum(ndcg_values) / max(num_q, 1), 4),
        "num_queries": num_q,
        "k": k,
        "per_query": per_query,
    }
