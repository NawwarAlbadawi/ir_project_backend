"""
=============================================================================
 Indexing Service — tfidf_indexer.py
=============================================================================
 Builds a TF-IDF matrix using scikit-learn's TfidfVectorizer.

 The pre-tokenized strings (from the Preprocessing Service) are fed
 directly to the vectorizer with a dummy tokenizer (identity function)
 so no additional tokenization is applied.

 Outputs:
   - A sparse CSR matrix (docs × terms)
   - The fitted vectorizer (to transform query strings)
   - Saved as pickle to disk for fast reloading
=============================================================================
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

logger = logging.getLogger("indexing_service.tfidf_indexer")


# ---------------------------------------------------------------------------
# Dummy tokenizer — documents are already preprocessed into token strings
# ---------------------------------------------------------------------------
def _identity_tokenizer(text: str) -> list[str]:
    return text.split()


class TFIDFIndexer:
    """
    Wraps scikit-learn TfidfVectorizer with disk persistence.

    After calling ``build()``, the object exposes:
      - ``matrix``: sparse (n_docs × n_terms) TF-IDF matrix
      - ``vectorizer``: fitted TfidfVectorizer (use to transform queries)
      - ``doc_ids``: ordered list of document IDs matching matrix rows
    """

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix: Optional[sp.csr_matrix] = None
        self.doc_ids: list[str] = []

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, documents: list[dict]) -> None:
        """
        Fit the TF-IDF model and transform all documents.

        Parameters
        ----------
        documents : list[dict]
            Each dict: {"doc_id": str, "cleaned": str, ...}
            ``cleaned`` is the space-joined preprocessed token string.
        """
        logger.info(f"Building TF-IDF index for {len(documents)} documents…")
        self.doc_ids = [d["doc_id"] for d in documents]
        corpus = [d["cleaned"] for d in documents]

        self.vectorizer = TfidfVectorizer(
            tokenizer=_identity_tokenizer,
            preprocessor=None,
            lowercase=False,       # already lowercased
            token_pattern=None,    # custom tokenizer takes over
            sublinear_tf=True,     # log(1+tf) — improves large corpora
            max_df=0.95,           # ignore terms in >95% of docs
            min_df=2,              # ignore terms in <2 docs
        )
        self.matrix = self.vectorizer.fit_transform(corpus)
        logger.info(
            f"TF-IDF index built: shape={self.matrix.shape}, "
            f"nnz={self.matrix.nnz}"
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def transform_query(self, cleaned_query: str) -> sp.csr_matrix:
        """
        Transform a preprocessed query string into a TF-IDF vector.

        Returns a sparse (1 × n_terms) matrix.
        """
        if self.vectorizer is None:
            raise RuntimeError("TF-IDF index not built. Call build() first.")
        return self.vectorizer.transform([cleaned_query])

    def search(self, cleaned_query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Retrieve top-K documents by cosine similarity to the query.

        Returns
        -------
        list of (doc_id, score) tuples, sorted descending by score.
        """
        q_vec = self.transform_query(cleaned_query)
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.doc_ids[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer, "matrix": self.matrix, "doc_ids": self.doc_ids},
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(f"TF-IDF index saved → {path}")

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        self.doc_ids = data["doc_ids"]
        logger.info(f"TF-IDF index loaded ← {path}  (shape={self.matrix.shape})")
        return True
