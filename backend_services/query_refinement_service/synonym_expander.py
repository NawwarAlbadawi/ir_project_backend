import logging
from typing import Optional
logger = logging.getLogger('query_refinement_service.synonym_expander')
try:
    import nltk
    from nltk.corpus import wordnet, stopwords
    from nltk import pos_tag, word_tokenize
    _nltk_available = True
except ImportError:
    _nltk_available = False
    logger.warning('NLTK not available — synonym expansion disabled.')
_PENN_TO_WN = {'J': 'a', 'V': 'v', 'N': 'n', 'R': 'r'}

class SynonymExpander:

    def __init__(self):
        if _nltk_available:
            self._stop_words = set(stopwords.words('english'))
        else:
            self._stop_words = set()

    def expand(self, query: str, max_expansions: int=3) -> tuple[str, dict[str, list[str]]]:
        if not _nltk_available:
            return (query, {})
        tokens = word_tokenize(query.lower())
        tagged = pos_tag(tokens)
        expansions: dict[str, list[str]] = {}
        extra_terms: list[str] = []
        for word, tag in tagged:
            if word in self._stop_words or len(word) < 3:
                continue
            wn_pos = _PENN_TO_WN.get(tag[0].upper(), 'n')
            synsets = wordnet.synsets(word, pos=wn_pos)
            if not synsets:
                continue
            synonyms: list[str] = []
            for synset in synsets[:2]:
                for lemma in synset.lemmas():
                    syn = lemma.name().replace('_', ' ').lower()
                    if syn != word and syn not in self._stop_words and (syn not in synonyms) and syn.isalpha():
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
            expanded = query + ' ' + ' '.join(extra_terms)
        return (expanded, expansions)