"""
=============================================================================
 Query Refinement Service — synonym_expander.py
=============================================================================
 Expands query terms with synonyms from WordNet (NLTK).

 Strategy:
   - For each content word in the query, fetch the top N lemma names
     from the first synset of the most likely POS.
   - Filter out the original word itself and common stop-words.
   - Append expansions to the query string (weighted by appending
     the synonyms once — query likelihood weighting).
=============================================================================
"""

import logging
from typing import Optional

logger = logging.getLogger("query_refinement_service.synonym_expander")

try:
    import nltk
    from nltk.corpus import wordnet, stopwords
    from nltk import pos_tag, word_tokenize
    _nltk_available = True
except ImportError:
    _nltk_available = False
    logger.warning("NLTK not available — synonym expansion disabled.")


_PENN_TO_WN = {
    "J": "a",  # adjective
    "V": "v",  # verb
    "N": "n",  # noun
    "R": "r",  # adverb
}


class SynonymExpander:
    """
    Expands each query token with WordNet synonyms.
    """

    def __init__(self):
        if _nltk_available:
            self._stop_words = set(stopwords.words("english"))
        else:
            self._stop_words = set()

    def expand(
        self,
        query: str,
        max_expansions: int = 3,
    ) -> tuple[str, dict[str, list[str]]]:
        """
        Expand the query with synonyms.

        Parameters
        ----------
        query : str
            The (possibly spell-corrected) query string.
        max_expansions : int
            Maximum number of synonyms to add per term.

        Returns
        -------
        expanded_query : str
        expansions : dict[str, list[str]]
            {original_term: [synonym1, synonym2, ...]}
        """
        if not _nltk_available:
            return query, {}

        tokens = word_tokenize(query.lower())
        tagged = pos_tag(tokens)
        expansions: dict[str, list[str]] = {}
        extra_terms: list[str] = []

        for word, tag in tagged:
            # Skip stop-words and very short words
            if word in self._stop_words or len(word) < 3:
                continue

            wn_pos = _PENN_TO_WN.get(tag[0].upper(), "n")
            synsets = wordnet.synsets(word, pos=wn_pos)
            if not synsets:
                continue

            synonyms: list[str] = []
            for synset in synsets[:2]:  # check top 2 synsets
                for lemma in synset.lemmas():
                    syn = lemma.name().replace("_", " ").lower()
                    if (
                        syn != word
                        and syn not in self._stop_words
                        and syn not in synonyms
                        and syn.isalpha()
                    ):
                        synonyms.append(syn)
                    if len(synonyms) >= max_expansions:
                        break
                if len(synonyms) >= max_expansions:
                    break

            if synonyms:
                expansions[word] = synonyms
                extra_terms.extend(synonyms)

        expanded = query
        if extra_terms:
            expanded = query + " " + " ".join(extra_terms)

        return expanded, expansions
