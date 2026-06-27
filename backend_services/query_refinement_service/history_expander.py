import logging
from collections import Counter
from typing import Optional
logger = logging.getLogger('query_refinement_service.history_expander')

class HistoryExpander:

    def __init__(self):
        self._history: dict[str, list[dict]] = {}

    def add_entry(self, session_id: str, query: str, clicked_doc_ids: Optional[list[str]]=None) -> None:
        tokens = query.lower().split()
        entry = {'query': query, 'tokens': tokens, 'clicked': clicked_doc_ids or []}
        if session_id not in self._history:
            self._history[session_id] = []
        self._history[session_id].append(entry)
        if len(self._history[session_id]) > 50:
            self._history[session_id] = self._history[session_id][-50:]

    def clear(self, session_id: str) -> None:
        self._history.pop(session_id, None)

    def expand(self, session_id: str, query: str, max_boosts: int=5) -> tuple[str, list[str]]:
        history = self._history.get(session_id, [])
        if not history:
            return (query, [])
        current_tokens = set(query.lower().split())
        term_weights: Counter = Counter()
        for entry in history:
            past_tokens = set(entry['tokens'])
            overlap = len(current_tokens & past_tokens)
            if overlap == 0:
                continue
            click_boost = 1.5 if entry['clicked'] else 1.0
            weight = overlap * click_boost
            new_terms = past_tokens - current_tokens
            for term in new_terms:
                if len(term) > 2:
                    term_weights[term] += weight
        if not term_weights:
            return (query, [])
        boosts = [term for term, _ in term_weights.most_common(max_boosts)]
        expanded = query + ' ' + ' '.join(boosts)
        logger.info(f'[history] Injected {len(boosts)} terms: {boosts}')
        return (expanded, boosts)