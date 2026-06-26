"""
=============================================================================
 Query Refinement Service — spell_corrector.py
=============================================================================
 Spelling correction using `pyspellchecker` (pure Python, no C deps).
 Falls back to no-op if the library is not installed.
=============================================================================
"""

import logging
from typing import Optional

logger = logging.getLogger("query_refinement_service.spell_corrector")

try:
    from spellchecker import SpellChecker as _SpellChecker
    _spellchecker_available = True
except ImportError:
    logger.warning("pyspellchecker not installed — spell correction disabled.")
    _spellchecker_available = False


class SpellCorrector:
    """
    Word-level spelling corrector.

    Uses edit-distance based candidate ranking from the pyspellchecker
    library.  Words already in the dictionary are passed through unchanged.
    """

    def __init__(self):
        if _spellchecker_available:
            self._spell = _SpellChecker()
        else:
            self._spell = None

    def correct(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Correct spelling in ``text``.

        Returns
        -------
        corrected_text : str
        corrections : dict[str, str]
            Mapping of {original_word: corrected_word} for changed words.
        """
        if self._spell is None:
            return text, {}

        words = text.split()
        corrections: dict[str, str] = {}
        corrected_words: list[str] = []

        # pyspellchecker can check a batch efficiently
        misspelled = self._spell.unknown(words)

        for word in words:
            if word.lower() in misspelled:
                candidate = self._spell.correction(word.lower())
                if candidate and candidate != word.lower():
                    corrections[word] = candidate
                    corrected_words.append(candidate)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        return " ".join(corrected_words), corrections
