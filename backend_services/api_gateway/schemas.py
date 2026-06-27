from pydantic import BaseModel, Field
from typing import Optional, Literal

class SearchRequest(BaseModel):
    query: str = Field(..., description='Raw query string from the user.')
    dataset: str = Field(..., description="'quora' or 'msmarco'")
    model: Literal['tfidf', 'bm25', 'word2vec', 'bert', 'hybrid_serial', 'hybrid_parallel'] = Field('bm25')
    top_k: int = Field(10, ge=1, le=1000)
    use_refinement: bool = Field(False)
    refinement_techniques: list[str] = Field(default=['spell', 'synonyms'])
    session_id: str = Field('default')
    bm25_k1: float = Field(1.5, ge=0.0, le=5.0)
    bm25_b: float = Field(0.75, ge=0.0, le=1.0)
    fusion_method: Literal['rrf', 'linear', 'combmnz'] = 'rrf'
    hybrid_weights: dict[str, float] = Field(default={'tfidf': 0.3, 'bm25': 0.4, 'bert': 0.3})
    serial_candidate_k: int = Field(100, ge=10, le=2000)
    preprocess_stem: bool = True
    preprocess_lemmatize: bool = False

class SearchResultItem(BaseModel):
    rank: int
    doc_id: str
    score: float
    snippet: Optional[str] = None

class SearchResponse(BaseModel):
    query_original: str
    query_refined: Optional[str] = None
    query_cleaned: str
    dataset: str
    model: str
    results: list[SearchResultItem]
    latency_ms: float
    refinement_info: Optional[dict] = None

class LoadDatasetRequest(BaseModel):
    dataset_name: str

class BuildIndexRequest(BaseModel):
    dataset_name: str
    models: list[str] = Field(default=['tfidf', 'bm25', 'word2vec', 'bert'])

class EvalCompareRequest(BaseModel):
    dataset: str
    model: str
    num_queries: int = Field(100, ge=1, le=500000, description='Number of queries to evaluate.')
    k: int = Field(10)
    compare_refinement: bool = Field(True, description='If True, evaluate both with and without refinement.')