"""
=============================================================================
 Query Refinement Service — main.py
 Port: 8005
=============================================================================
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query as QParam
from fastapi.middleware.cors import CORSMiddleware

from query_refinement_service.schemas import RefineRequest, RefineResponse, HistoryEntry
from query_refinement_service.spell_corrector import SpellCorrector
from query_refinement_service.synonym_expander import SynonymExpander
from query_refinement_service.history_expander import HistoryExpander

logger = logging.getLogger("query_refinement_service")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

spell_corrector: SpellCorrector = None
synonym_expander: SynonymExpander = None
history_expander: HistoryExpander = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global spell_corrector, synonym_expander, history_expander
    logger.info("Query Refinement Service starting…")
    spell_corrector = SpellCorrector()
    synonym_expander = SynonymExpander()
    history_expander = HistoryExpander()
    logger.info("Query Refinement Service ready.")
    yield


app = FastAPI(
    title="Query Refinement Service",
    description="Spelling correction, synonym expansion, and history-based query expansion.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "service": "query_refinement"}


@app.post("/refine", response_model=RefineResponse, tags=["refinement"])
def refine_query(request: RefineRequest):
    """
    Apply one or more refinement techniques to the input query.

    Techniques are applied in this order:
      1. spell       — correct misspellings
      2. synonyms    — expand with WordNet synonyms
      3. history     — inject terms from session history

    The refined query is the cumulative result of all active techniques.
    """
    query = request.query
    corrections: dict[str, str] = {}
    expansions: dict[str, list[str]] = {}
    history_boosts: list[str] = []
    applied: list[str] = []

    # 1. Spell correction
    if "spell" in request.techniques:
        query, corrections = spell_corrector.correct(query)
        if corrections:
            applied.append("spell")

    # 2. Synonym expansion
    if "synonyms" in request.techniques:
        query, expansions = synonym_expander.expand(
            query, max_expansions=request.max_expansions
        )
        if expansions:
            applied.append("synonyms")

    # 3. History-based expansion
    if "history" in request.techniques:
        query, history_boosts = history_expander.expand(
            session_id=request.session_id,
            query=query,
            max_boosts=request.max_expansions,
        )
        if history_boosts:
            applied.append("history")

    return RefineResponse(
        original_query=request.query,
        refined_query=query,
        corrections=corrections,
        expansions=expansions,
        history_boosts=history_boosts,
        techniques_applied=applied,
    )


@app.post("/refine/history", tags=["history"])
def add_to_history(entry: HistoryEntry):
    """
    Record a query (with optional click signals) into the session history.

    Call this after every successful search so the history expander
    learns from the session.
    """
    history_expander.add_entry(
        session_id=entry.session_id,
        query=entry.query,
        clicked_doc_ids=entry.clicked_doc_ids,
    )
    return {"message": "History entry recorded."}


@app.delete("/refine/history", tags=["history"])
def clear_history(session_id: str = QParam("default")):
    """Clear the query history for a session."""
    history_expander.clear(session_id)
    return {"message": f"History cleared for session '{session_id}'."}
