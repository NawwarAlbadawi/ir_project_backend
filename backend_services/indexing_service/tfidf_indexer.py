import logging
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
logger = logging.getLogger('indexing_service.tfidf_indexer')

def _identity_tokenizer(text: str) -> list[str]:
    return text.split()

class TFIDFIndexer:

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix: Optional[sp.csr_matrix] = None
        self.doc_ids: list[str] = []

    def build(self, documents: list[dict]) -> None:
        logger.info(f'Building TF-IDF index for {len(documents)} documents…')
        self.doc_ids = [d['doc_id'] for d in documents]
        corpus = [d['cleaned'] for d in documents]
        self.vectorizer = TfidfVectorizer(tokenizer=_identity_tokenizer, preprocessor=None, lowercase=False, token_pattern=None, sublinear_tf=True, max_df=0.95, min_df=2)
        self.matrix = self.vectorizer.fit_transform(corpus)
        logger.info(f'TF-IDF index built: shape={self.matrix.shape}, nnz={self.matrix.nnz}')

    def transform_query(self, cleaned_query: str) -> sp.csr_matrix:
        if self.vectorizer is None:
            raise RuntimeError('TF-IDF index not built. Call build() first.')
        return self.vectorizer.transform([cleaned_query])

    def search(self, cleaned_query: str, top_k: int=10) -> list[tuple[str, float]]:
        q_vec = self.transform_query(cleaned_query)
        scores = cosine_similarity(q_vec, self.matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'matrix': self.matrix, 'doc_ids': self.doc_ids}, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f'TF-IDF index saved → {path}')

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.vectorizer = data['vectorizer']
        self.matrix = data['matrix']
        self.doc_ids = data['doc_ids']
        logger.info(f'TF-IDF index loaded ← {path}  (shape={self.matrix.shape})')
        return True