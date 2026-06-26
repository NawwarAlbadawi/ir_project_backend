"""
=============================================================================
 Indexing Service — inverted_index.py
=============================================================================
 Builds and persists a classic Inverted Index.

 Structure:
   index[term] = {
       "df": int,                        # document frequency
       "postings": {doc_id: tf, ...}     # term frequency per doc
   }

 Also stores:
   - doc_lengths: {doc_id: total_term_count}  (for BM25 / normalization)
   - avg_doc_length: float
   - N: total number of documents
=============================================================================
"""

import json
import logging
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger("indexing_service.inverted_index")


class InvertedIndex:
    """
    Classic Inverted Index built from token lists.

    Serialized to disk as a gzipped pickle for fast reload.
    """

    def __init__(self):
        # term → {"df": int, "postings": {doc_id: tf}}
        self._index: dict[str, dict] = defaultdict(lambda: {"df": 0, "postings": {}})
        self._doc_lengths: dict[str, int] = {}  # doc_id → token count
        self._doc_ids: list[str] = []           # ordered
        self.N: int = 0
        self.avg_doc_length: float = 0.0

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self, documents: list[dict]) -> None:
        """
        Build the index from a list of preprocessed documents.

        Parameters
        ----------
        documents : list[dict]
            Each dict: {"doc_id": str, "tokens": list[str], ...}
        """
        logger.info(f"Building inverted index for {len(documents)} documents…")
        self._index.clear()
        self._doc_lengths.clear()
        self._doc_ids.clear()

        total_tokens = 0
        for doc in documents:
            doc_id = doc["doc_id"]
            tokens: list[str] = doc["tokens"]
            self._doc_ids.append(doc_id)

            # Count term frequencies within this document
            tf_counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                tf_counts[token] += 1

            # Update index
            for term, tf in tf_counts.items():
                entry = self._index[term]
                entry["postings"][doc_id] = tf
                entry["df"] = len(entry["postings"])

            doc_len = len(tokens)
            self._doc_lengths[doc_id] = doc_len
            total_tokens += doc_len

        self.N = len(documents)
        self.avg_doc_length = total_tokens / max(self.N, 1)

        logger.info(
            f"Inverted index built: {self.vocab_size} unique terms, "
            f"avg_doc_len={self.avg_doc_length:.1f}"
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get_postings(self, term: str) -> dict[str, int]:
        """Return {doc_id: tf} for a term, or {} if not found."""
        entry = self._index.get(term)
        return entry["postings"] if entry else {}

    def get_df(self, term: str) -> int:
        entry = self._index.get(term)
        return entry["df"] if entry else 0

    def get_doc_length(self, doc_id: str) -> int:
        return self._doc_lengths.get(doc_id, 0)

    @property
    def vocab_size(self) -> int:
        return len(self._index)

    @property
    def doc_ids(self) -> list[str]:
        return self._doc_ids

    def term_info(self, term: str) -> Optional[dict]:
        """Return {"df": int, "postings": dict} or None."""
        return self._index.get(term)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "index": dict(self._index),
            "doc_lengths": self._doc_lengths,
            "doc_ids": self._doc_ids,
            "N": self.N,
            "avg_doc_length": self.avg_doc_length,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Inverted index saved → {path}")

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._index = defaultdict(lambda: {"df": 0, "postings": {}}, data["index"])
        self._doc_lengths = data["doc_lengths"]
        self._doc_ids = data["doc_ids"]
        self.N = data["N"]
        self.avg_doc_length = data["avg_doc_length"]
        logger.info(f"Inverted index loaded ← {path}  ({self.vocab_size} terms, {self.N} docs)")
        return True
