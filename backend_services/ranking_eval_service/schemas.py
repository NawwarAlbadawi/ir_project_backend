from pydantic import BaseModel, Field
from typing import Optional

class EvaluateRequest(BaseModel):
    dataset: str
    model: str
    query_id: str
    results: list[dict] = Field(..., description='List of {doc_id, score, rank} dicts from retrieval.')
    qrel: dict[str, int] = Field(..., description='Relevance judgments: {doc_id: relevance_score}')
    k: int = Field(10, description='Cut-off for P@k and nDCG@k')

class PerQueryMetrics(BaseModel):
    query_id: str
    average_precision: float
    recall: float
    precision_at_k: float
    ndcg_at_k: float

class FullEvalRequest(BaseModel):
    dataset: str
    model: str
    ranked_results: dict[str, list[dict]] = Field(..., description='query_id → list of {doc_id, score, rank}')
    qrels: dict[str, dict[str, int]] = Field(..., description='query_id → {doc_id: relevance}')
    k: int = Field(10)

class AggregateMetrics(BaseModel):
    dataset: str
    model: str
    map: float = Field(..., description='Mean Average Precision')
    mean_recall: float
    mean_precision_at_k: float
    mean_ndcg_at_k: float
    num_queries: int
    k: int
    per_query: Optional[list[PerQueryMetrics]] = None