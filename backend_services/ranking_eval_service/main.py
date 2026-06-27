import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ranking_eval_service.schemas import EvaluateRequest, PerQueryMetrics, FullEvalRequest, AggregateMetrics
from ranking_eval_service.evaluator import average_precision, recall_at_k, precision_at_k, ndcg_at_k, evaluate_all
logger = logging.getLogger('ranking_eval_service')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
app = FastAPI(title='Ranking & Evaluation Service', description='Computes MAP, Recall, P@K, nDCG@K for retrieval results.', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

@app.get('/health', tags=['meta'])
def health():
    return {'status': 'ok', 'service': 'ranking_eval'}

@app.post('/evaluate', response_model=PerQueryMetrics, tags=['evaluation'])
def evaluate_single(request: EvaluateRequest):
    ranked_ids = [r['doc_id'] for r in request.results]
    return PerQueryMetrics(query_id=request.query_id, average_precision=round(average_precision(ranked_ids, request.qrel), 4), recall=round(recall_at_k(ranked_ids, request.qrel, request.k), 4), precision_at_k=round(precision_at_k(ranked_ids, request.qrel, request.k), 4), ndcg_at_k=round(ndcg_at_k(ranked_ids, request.qrel, request.k), 4))

@app.post('/evaluate/full', response_model=AggregateMetrics, tags=['evaluation'])
def evaluate_full(request: FullEvalRequest):
    ranked_results: dict[str, list[str]] = {qid: [r['doc_id'] for r in results] for qid, results in request.ranked_results.items()}
    metrics = evaluate_all(ranked_results, request.qrels, k=request.k)
    return AggregateMetrics(dataset=request.dataset, model=request.model, map=metrics['map'], mean_recall=metrics['mean_recall'], mean_precision_at_k=metrics['mean_precision_at_k'], mean_ndcg_at_k=metrics['mean_ndcg_at_k'], num_queries=metrics['num_queries'], k=metrics['k'], per_query=[PerQueryMetrics(**pq) for pq in metrics['per_query']])