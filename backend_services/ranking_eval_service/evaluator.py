import logging
import math
from typing import Optional
logger = logging.getLogger('ranking_eval_service.evaluator')

def average_precision(ranked_doc_ids: list[str], relevant_docs: dict[str, int], relevance_threshold: int=1) -> float:
    R = sum((1 for v in relevant_docs.values() if v >= relevance_threshold))
    if R == 0:
        return 0.0
    num_relevant_found = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if relevant_docs.get(doc_id, 0) >= relevance_threshold:
            num_relevant_found += 1
            precision_sum += num_relevant_found / rank
    return precision_sum / R

def recall_at_k(ranked_doc_ids: list[str], relevant_docs: dict[str, int], k: int, relevance_threshold: int=1) -> float:
    R = sum((1 for v in relevant_docs.values() if v >= relevance_threshold))
    if R == 0:
        return 0.0
    top_k = ranked_doc_ids[:k]
    retrieved_relevant = sum((1 for d in top_k if relevant_docs.get(d, 0) >= relevance_threshold))
    return retrieved_relevant / R

def precision_at_k(ranked_doc_ids: list[str], relevant_docs: dict[str, int], k: int, relevance_threshold: int=1) -> float:
    top_k = ranked_doc_ids[:k]
    if not top_k:
        return 0.0
    relevant = sum((1 for d in top_k if relevant_docs.get(d, 0) >= relevance_threshold))
    return relevant / k

def dcg_at_k(ranked_doc_ids: list[str], relevant_docs: dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = relevant_docs.get(doc_id, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg

def ideal_dcg_at_k(relevant_docs: dict[str, int], k: int) -> float:
    sorted_rels = sorted(relevant_docs.values(), reverse=True)
    idcg = 0.0
    for i, rel in enumerate(sorted_rels[:k], start=1):
        if rel > 0:
            idcg += (2 ** rel - 1) / math.log2(i + 1)
    return idcg

def ndcg_at_k(ranked_doc_ids: list[str], relevant_docs: dict[str, int], k: int) -> float:
    idcg = ideal_dcg_at_k(relevant_docs, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_doc_ids, relevant_docs, k) / idcg

def evaluate_all(ranked_results: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int=10, relevance_threshold: int=1) -> dict:
    per_query = []
    ap_values, recall_values, p_values, ndcg_values = ([], [], [], [])
    for query_id, doc_ids in ranked_results.items():
        rel = qrels.get(query_id, {})
        if not rel:
            logger.debug(f'No qrels for query {query_id}, skipping.')
            continue
        ap = average_precision(doc_ids, rel, relevance_threshold)
        rec = recall_at_k(doc_ids, rel, k, relevance_threshold)
        prec = precision_at_k(doc_ids, rel, k, relevance_threshold)
        ndcg = ndcg_at_k(doc_ids, rel, k)
        ap_values.append(ap)
        recall_values.append(rec)
        p_values.append(prec)
        ndcg_values.append(ndcg)
        per_query.append({'query_id': query_id, 'average_precision': round(ap, 4), 'recall': round(rec, 4), 'precision_at_k': round(prec, 4), 'ndcg_at_k': round(ndcg, 4)})
    num_q = len(ap_values)
    return {'map': round(sum(ap_values) / max(num_q, 1), 4), 'mean_recall': round(sum(recall_values) / max(num_q, 1), 4), 'mean_precision_at_k': round(sum(p_values) / max(num_q, 1), 4), 'mean_ndcg_at_k': round(sum(ndcg_values) / max(num_q, 1), 4), 'num_queries': num_q, 'k': k, 'per_query': per_query}