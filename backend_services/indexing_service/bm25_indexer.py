import logging
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi
logger = logging.getLogger('indexing_service.bm25_indexer')
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

class BM25Indexer:

    def __init__(self):
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: list[list[str]] = []
        self.doc_ids: list[str] = []
        self._k1: float = DEFAULT_K1
        self._b: float = DEFAULT_B

    def build(self, documents: list[dict], k1: float=DEFAULT_K1, b: float=DEFAULT_B) -> None:
        logger.info(f'Building BM25 index for {len(documents)} documents (k1={k1}, b={b})…')
        self.doc_ids = [d['doc_id'] for d in documents]
        self._tokenized_corpus = [d['tokens'] for d in documents]
        self._k1 = k1
        self._b = b
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=k1, b=b)
        logger.info('BM25 index built.')

    def retune(self, k1: float, b: float) -> None:
        if not self._tokenized_corpus:
            raise RuntimeError('BM25 index not built. Call build() first.')
        logger.info(f'Retuning BM25: k1={k1}, b={b}')
        self._k1 = k1
        self._b = b
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=k1, b=b)

    def search(self, query_tokens: list[str], top_k: int=10, k1: Optional[float]=None, b: Optional[float]=None) -> list[tuple[str, float]]:
        if self._bm25 is None:
            raise RuntimeError('BM25 index not built.')
        if k1 is not None and k1 != self._k1 or (b is not None and b != self._b):
            self.retune(k1 or self._k1, b or self._b)
        scores: np.ndarray = self._bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'tokenized_corpus': self._tokenized_corpus, 'doc_ids': self.doc_ids, 'k1': self._k1, 'b': self._b}, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f'BM25 index saved → {path}')

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._tokenized_corpus = data['tokenized_corpus']
        self.doc_ids = data['doc_ids']
        self._k1 = data['k1']
        self._b = data['b']
        self._bm25 = BM25Okapi(self._tokenized_corpus, k1=self._k1, b=self._b)
        logger.info(f'BM25 index loaded ← {path}  ({len(self.doc_ids)} docs, k1={self._k1}, b={self._b})')
        return True