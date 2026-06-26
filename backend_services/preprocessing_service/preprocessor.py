"""
=============================================================================
 Preprocessing Service — preprocessor.py
=============================================================================
 Core preprocessing logic:
   1. Lowercase normalization
   2. Punctuation removal
   3. Tokenization (NLTK word_tokenize)
   4. Stop-word removal
   5. Lemmatization (WordNetLemmatizer) — optional
   6. Stemming (PorterStemmer)           — optional

 Both lemmatization and stemming may be applied; if both are selected,
 lemmatization runs first (it is a more principled reduction).
=============================================================================
"""

import logging
import re
import string

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from preprocessing_service.schemas import PreprocessOptions

logger = logging.getLogger("preprocessing_service.preprocessor")


def _ensure_nltk_data():
    """Download required NLTK corpora exactly once."""
    packages = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, pkg in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            logger.info(f"Downloading NLTK package: {pkg}")
            nltk.download(pkg, quiet=True)


# ---------------------------------------------------------------------------
# POS tag helper for better lemmatization
# ---------------------------------------------------------------------------
_WN_POS_MAP = {
    "J": wordnet.ADJ,
    "V": wordnet.VERB,
    "N": wordnet.NOUN,
    "R": wordnet.ADV,
}


def _penn_to_wn(tag: str) -> str:
    """Convert a Penn Treebank POS tag to a WordNet POS tag."""
    return _WN_POS_MAP.get(tag[0].upper(), wordnet.NOUN)


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------
class Preprocessor:
    """
    Stateless (after initialisation) text processor.

    Thread-safe: all state is initialised once and never mutated at
    call time, so instances can be shared across async workers.
    """

    def __init__(self):
        _ensure_nltk_data()
        self._stop_words: set[str] = set(stopwords.words("english"))
        self._stemmer = PorterStemmer()
        self._lemmatizer = WordNetLemmatizer()
        # Pre-build a translation table that maps every punctuation char → None
        self._punct_table = str.maketrans("", "", string.punctuation)
        logger.info("Preprocessor initialised (stemmer + lemmatizer ready).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def preprocess(self, text: str, options: PreprocessOptions | None = None) -> list[str]:
        """
        Apply the full preprocessing pipeline to *text* and return a list
        of cleaned tokens.

        Parameters
        ----------
        text : str
            Raw input text.
        options : PreprocessOptions
            Which steps to apply. Uses defaults if None.

        Returns
        -------
        list[str]
            Ordered list of processed tokens (stop-words / empties removed).
        """
        if options is None:
            options = PreprocessOptions()

        # 1. Lowercase
        if options.lowercase:
            text = text.lower()

        # 2. Remove URLs and other noise before punctuation stripping
        text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
        text = re.sub(r"\d+", " ", text)                 # digits (optional debate)

        # 3. Punctuation removal
        if options.remove_punctuation:
            text = text.translate(self._punct_table)

        # 4. Tokenise
        tokens: list[str] = word_tokenize(text)

        # 5. Stop-word removal (also removes single-character tokens)
        if options.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words and len(t) > 1]

        # 6. Lemmatize (uses POS tags for accuracy)
        if options.lemmatize:
            tokens = self._lemmatize_tokens(tokens)

        # 7. Stem
        if options.stem:
            tokens = [self._stemmer.stem(t) for t in tokens]

        # Remove any empty strings produced by earlier steps
        tokens = [t for t in tokens if t.strip()]
        return tokens

    def preprocess_to_string(self, text: str, options: PreprocessOptions | None = None) -> str:
        """Convenience wrapper — returns tokens joined by spaces."""
        return " ".join(self.preprocess(text, options))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        """
        Lemmatize a list of tokens using POS-aware lemmatization.

        POS tagging improves lemmatization quality significantly:
        e.g., "running" → verb → "run" (not "running").
        """
        tagged = pos_tag(tokens)
        return [
            self._lemmatizer.lemmatize(word, pos=_penn_to_wn(tag))
            for word, tag in tagged
        ]
