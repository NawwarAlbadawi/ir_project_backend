"""
=============================================================================
 Topic Service — schemas.py
=============================================================================
"""
from pydantic import BaseModel, Field
from typing import Optional


class BuildTopicsRequest(BaseModel):
    dataset_name: str
    num_topics: int = Field(10, ge=2, le=50, description="Number of LDA topics")
    num_top_words: int = Field(8, ge=3, le=20, description="Top words per topic")


class TopicDefinition(BaseModel):
    topic_id: int
    label: str                  # auto-generated label from top words
    top_words: list[str]
    doc_count: int              # number of docs assigned to this topic


class BuildTopicsResponse(BaseModel):
    dataset: str
    status: str
    num_topics: int
    topics: list[TopicDefinition] = []
    error: Optional[str] = None


class DocTopicRequest(BaseModel):
    doc_ids: list[str]
    dataset_name: str


class DocTopicResult(BaseModel):
    doc_id: str
    topic_id: int
    topic_label: str
    top_words: list[str]
    probability: float          # probability of the dominant topic


class DocTopicResponse(BaseModel):
    dataset: str
    results: list[DocTopicResult]


class TopicStatusResponse(BaseModel):
    dataset: str
    status: str                 # not_built | building | ready | error
    num_topics: int = 0
    error: Optional[str] = None
