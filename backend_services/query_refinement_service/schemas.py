"""
=============================================================================
 Query Refinement Service — schemas.py
=============================================================================
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class RefineRequest(BaseModel):
    query: str = Field(..., description="Raw query string from the user.")
    techniques: list[Literal["spell", "synonyms", "history"]] = Field(
        default=["spell", "synonyms"],
        description="Which refinement techniques to apply.",
    )
    session_id: str = Field("default", description="Session key for history-based expansion.")
    max_expansions: int = Field(3, ge=1, le=10, description="Max synonyms/expansions per term.")


class RefineResponse(BaseModel):
    original_query: str
    refined_query: str
    corrections: dict[str, str] = Field(
        default_factory=dict,
        description="Spelling corrections applied: {original_word: corrected_word}",
    )
    expansions: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Synonym expansions: {term: [synonyms]}",
    )
    history_boosts: list[str] = Field(
        default_factory=list,
        description="Terms injected from query history.",
    )
    techniques_applied: list[str] = []


class HistoryEntry(BaseModel):
    session_id: str = "default"
    query: str
    clicked_doc_ids: list[str] = Field(default_factory=list)
