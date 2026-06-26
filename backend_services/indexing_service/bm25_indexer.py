"""
=============================================================================
 Indexing Service — bm25_indexer.py
=============================================================================
 Wraps the `rank-bm25` library (BM25Okapi) with:
   - Tunable k1 and b parameters (settable at retrieval time)
   - Disk persistence via pickle
   - Search returning (doc_id, score) tuples
=============================================================================
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger("indexing_service.bm25_indexer")

# BM25 Okapi defaults
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25Indexer:
    """
    BM25 index wrapping rank-bm25's BM25Okapi.

    ``k1`` and ``b`` are **retrieval-time** parameters; changing them
    does NOT require rebuilding the index — we rebuild the BM25 object
    from the stored token lists with the new parameters.
    """

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: list[list[str]] = []
        self.doc_ids: list[str] = []
        self._k1: float = DEFAULT_K1
        self._b: float = DEFAULT_B

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, documents: list[dict], k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        """
        Build the BM25 index from preprocessed documents.

        Parameters
        ----------
        documents : list[dict]
            Each dict: {"doc_id": str, "tokens": list[str], ...}
        k1 : float
            Term saturation parameter (typical range 1.2–2.0).
        b : float
            Length normalization parameter (0.0 = no normalization, 1.0 = full).
        """
        logger.info(f"Building BM25 index for {len(documents)} documents (k1={k1}, b={b})…")
        self.doc_ids = [d["doc_id"] for d in documents]
        self._tokenized_corpus = [d["tokens"] for d in documents]
        self._k1 = k1
        self._b = b
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=k1, b=b)
        logger.info("BM25 index built.")

    # ------------------------------------------------------------------
    # Parameter tuning (no full rebuild needed)
    # ------------------------------------------------------------------
    def retune(self, k1: float, b: float) -> None:
        """
        Re-create the BM25 object with new k1 / b parameters.

        This is O(N) but much faster than re-building from documents
        because we already have the tokenized corpus in memory.
        """
        if not self._tokenized_corpus:
            raise RuntimeError("BM25 index not built. Call build() first.")
        logger.info(f"Retuning BM25: k1={k1}, b={b}")
        self._k1 = k1
        self._b = b
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=k1, b=b)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query_tokens: list[str],
        top_k: int = 10,
        k1: Optional[float] = None,
        b: Optional[float] = None,
    ) -> list[tuple[str, float]]:
        """
        Retrieve top-K documents by BM25 score.

        If k1 / b are provided and differ from current values,
        the model is re-tuned before scoring.

        Returns
        -------
        list of (doc_id, score) tuples, sorted descending by score.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25 index not built.")

        # Re-tune on-the-fly if parameters changed
        if (k1 is not None and k1 != self._k1) or (b is not None and b != self._b):
            self.retune(k1 or self._k1, b or self._b)

        scores: np.ndarray = self._bm25.get_scores(query_tokens)
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
                {
                    "tokenized_corpus": self._tokenized_corpus,
                    "doc_ids": self.doc_ids,
                    "k1": self._k1,
                    "b": self._b,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        logger.info(f"BM25 index saved → {path}")

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._tokenized_corpus = data["tokenized_corpus"]
        self.doc_ids = data["doc_ids"]
        self._k1 = data["k1"]
        self._b = data["b"]
        # Reconstruct BM25 object
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self._k1, b=self._b)
        logger.info(
            f"BM25 index loaded ← {path}  "
            f"({len(self.doc_ids)} docs, k1={self._k1}, b={self._b})"
        )
        return True
