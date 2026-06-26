"""
=============================================================================
 Query Refinement Service — history_expander.py
=============================================================================
 Session-aware query expansion based on click history.

 Mechanism:
   - The service tracks past queries in each session.
   - When the user issues a new query, we find past queries that share
     terms with the current query.
   - Terms from those past queries are added to the current query,
     with higher weight given to queries that produced user clicks
     (i.e., positive relevance feedback).

 Storage: in-memory dict — resets on service restart.
 For production, swap the dict for Redis or a database.
=============================================================================
"""

import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger("query_refinement_service.history_expander")


class HistoryExpander:
    """
    Manages per-session query history and produces history-based expansions.
    """

    def __init__(self):
        # session_id → list of {"query": str, "tokens": list[str], "clicked": list[str]}
        self._history: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add_entry(
        self,
        session_id: str,
        query: str,
        clicked_doc_ids: Optional[list[str]] = None,
    ) -> None:
        """Record a query (and optional click signals) in the session history."""
        tokens = query.lower().split()
        entry = {
            "query": query,
            "tokens": tokens,
            "clicked": clicked_doc_ids or [],
        }
        if session_id not in self._history:
            self._history[session_id] = []
        self._history[session_id].append(entry)
        # Keep history bounded to avoid memory leaks
        if len(self._history[session_id]) > 50:
            self._history[session_id] = self._history[session_id][-50:]

    def clear(self, session_id: str) -> None:
        self._history.pop(session_id, None)

    def expand(
        self,
        session_id: str,
        query: str,
        max_boosts: int = 5,
    ) -> tuple[str, list[str]]:
        """
        Expand the query using session history.

        Parameters
        ----------
        session_id : str
        query : str
            Current (possibly already spell-corrected + synonym-expanded) query.
        max_boosts : int
            Maximum number of history-derived terms to inject.

        Returns
        -------
        expanded_query : str
        boosts : list[str]
            The terms that were injected.
        """
        history = self._history.get(session_id, [])
        if not history:
            return query, []

        current_tokens = set(query.lower().split())
        term_weights: Counter = Counter()

        for entry in history:
            past_tokens = set(entry["tokens"])
            # Overlap score: how many terms match the current query
            overlap = len(current_tokens & past_tokens)
            if overlap == 0:
                continue

            # Weight boost: entries with clicks are more valuable
            click_boost = 1.5 if entry["clicked"] else 1.0
            weight = overlap * click_boost

            # Collect new terms from this past query
            new_terms = past_tokens - current_tokens
            for term in new_terms:
                if len(term) > 2:   # Skip very short tokens
                    term_weights[term] += weight

        if not term_weights:
            return query, []

        # Select top-N terms by weight
        boosts = [term for term, _ in term_weights.most_common(max_boosts)]
        expanded = query + " " + " ".join(boosts)
        logger.info(f"[history] Injected {len(boosts)} terms: {boosts}")
        return expanded, boosts
