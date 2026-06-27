from pydantic import BaseModel, Field
from typing import Optional

class BuildIndexRequest(BaseModel):
    dataset_name: str = Field(..., description="'quora' or 'msmarco'")
    models: list[str] = Field(default=['tfidf', 'bm25', 'word2vec', 'bert'], description='Which indexes to build.')

class IndexStatusResponse(BaseModel):
    dataset: str
    status: str
    built_models: list[str] = []
    progress: int = 0
    total: int = 0
    error: Optional[str] = None

class InvertedIndexQueryResponse(BaseModel):
    term: str
    postings: dict[str, int]
    df: int

class IndexStatsResponse(BaseModel):
    dataset: str
    num_docs: int
    vocab_size: int
    available_models: list[str]