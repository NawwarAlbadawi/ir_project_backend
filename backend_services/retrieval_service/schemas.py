from pydantic import BaseModel, Field
from typing import Optional, Literal

class RetrievalRequest(BaseModel):
    query_tokens: list[str] = Field(..., description='Preprocessed query tokens.')
    query_cleaned: str = Field(..., description='Tokens joined by spaces.')
    query_raw: str = Field('', description='Original raw query (used by BERT).')
    dataset: str = Field(..., description="'quora' or 'msmarco'")
    model: Literal['tfidf', 'bm25', 'word2vec', 'bert', 'hybrid_serial', 'hybrid_parallel'] = Field('bm25')
    top_k: int = Field(10, ge=1, le=1000)
    bm25_k1: float = Field(1.5, ge=0.0, le=5.0)
    bm25_b: float = Field(0.75, ge=0.0, le=1.0)
    fusion_method: Literal['rrf', 'linear', 'combmnz'] = Field('rrf')
    hybrid_weights: dict[str, float] = Field(default={'tfidf': 0.3, 'bm25': 0.4, 'bert': 0.3}, description='Weights for linear fusion in hybrid_parallel.')
    serial_candidate_k: int = Field(100, ge=10, le=2000)

class RetrievalResult(BaseModel):
    doc_id: str
    score: float
    rank: int

class RetrievalResponse(BaseModel):
    query_raw: str
    query_cleaned: str
    dataset: str
    model: str
    results: list[RetrievalResult]
    latency_ms: float