import logging
import pickle
from pathlib import Path
from typing import Optional, Literal
import numpy as np
logger = logging.getLogger('indexing_service.embedding_indexer')
_gensim_available = False
_st_available = False
try:
    from gensim.models import Word2Vec
    _gensim_available = True
except ImportError:
    logger.warning('gensim not installed — Word2Vec unavailable.')
try:
    from sentence_transformers import SentenceTransformer
    _st_available = True
except ImportError:
    logger.warning('sentence-transformers not installed — BERT unavailable.')

def _cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    return matrix @ q_norm

class Word2VecIndexer:
    W2V_DIM = 200
    W2V_WINDOW = 5
    W2V_MIN_COUNT = 2
    W2V_EPOCHS = 5

    def __init__(self):
        self._model: Optional[object] = None
        self._matrix: Optional[np.ndarray] = None
        self.doc_ids: list[str] = []

    def build(self, documents: list[dict]) -> None:
        if not _gensim_available:
            raise RuntimeError('gensim is required for Word2Vec. pip install gensim')
        logger.info(f'Training Word2Vec on {len(documents)} documents…')
        self.doc_ids = [d['doc_id'] for d in documents]
        tokenized = [d['tokens'] for d in documents]
        self._model = Word2Vec(sentences=tokenized, vector_size=self.W2V_DIM, window=self.W2V_WINDOW, min_count=self.W2V_MIN_COUNT, workers=4, epochs=self.W2V_EPOCHS, seed=42)
        logger.info('Word2Vec trained. Building document vectors…')
        vectors = []
        wv = self._model.wv
        for tokens in tokenized:
            vecs = [wv[t] for t in tokens if t in wv]
            if vecs:
                avg = np.mean(vecs, axis=0).astype(np.float32)
            else:
                avg = np.zeros(self.W2V_DIM, dtype=np.float32)
            norm = np.linalg.norm(avg)
            vectors.append(avg / (norm + 1e-10))
        self._matrix = np.stack(vectors)
        logger.info(f'Word2Vec index built: shape={self._matrix.shape}')

    def query_vector(self, tokens: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('Word2Vec index not built.')
        wv = self._model.wv
        vecs = [wv[t] for t in tokens if t in wv]
        if vecs:
            avg = np.mean(vecs, axis=0).astype(np.float32)
        else:
            avg = np.zeros(self.W2V_DIM, dtype=np.float32)
        norm = np.linalg.norm(avg)
        return avg / (norm + 1e-10)

    def search(self, tokens: list[str], top_k: int=10) -> list[tuple[str, float]]:
        q = self.query_vector(tokens)
        scores = _cosine_scores(q, self._matrix)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(str(path / 'w2v.model'))
        np.save(str(path / 'w2v_matrix.npy'), self._matrix)
        with open(path / 'w2v_docids.pkl', 'wb') as f:
            pickle.dump(self.doc_ids, f)
        logger.info(f'Word2Vec index saved → {path}')

    def load(self, path: Path) -> bool:
        path = Path(path)
        model_path = path / 'w2v.model'
        matrix_path = path / 'w2v_matrix.npy'
        ids_path = path / 'w2v_docids.pkl'
        if not all((p.exists() for p in [model_path, matrix_path, ids_path])):
            return False
        if not _gensim_available:
            raise RuntimeError('gensim is required for Word2Vec.')
        self._model = Word2Vec.load(str(model_path))
        self._matrix = np.load(str(matrix_path))
        with open(ids_path, 'rb') as f:
            self.doc_ids = pickle.load(f)
        logger.info(f'Word2Vec index loaded ← {path}  (shape={self._matrix.shape})')
        return True

class BERTIndexer:
    MODEL_NAME = 'all-MiniLM-L6-v2'
    BATCH_SIZE = 256

    def __init__(self):
        self._model: Optional[object] = None
        self._matrix: Optional[np.ndarray] = None
        self.doc_ids: list[str] = []

    def build(self, documents: list[dict]) -> None:
        if not _st_available:
            raise RuntimeError('sentence-transformers is required for BERT indexing. pip install sentence-transformers')
        logger.info(f'Encoding {len(documents)} documents with {self.MODEL_NAME}… (this may take several minutes on CPU)')
        self.doc_ids = [d['doc_id'] for d in documents]
        texts = [d.get('raw_text') or d.get('cleaned', '') for d in documents]
        self._model = SentenceTransformer(self.MODEL_NAME)
        embeddings = self._model.encode(texts, batch_size=self.BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        self._matrix = embeddings
        logger.info(f'BERT index built: shape={self._matrix.shape}')

    def query_vector(self, raw_query: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError('BERT index not built.')
        vec = self._model.encode([raw_query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
        return vec

    def search(self, raw_query: str, top_k: int=10) -> list[tuple[str, float]]:
        q = self.query_vector(raw_query)
        scores = _cosine_scores(q, self._matrix)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path / 'bert_config.pkl', 'wb') as f:
            pickle.dump({'model_name': self.MODEL_NAME, 'doc_ids': self.doc_ids}, f)
        np.save(str(path / 'bert_matrix.npy'), self._matrix)
        logger.info(f'BERT index saved → {path}')

    def load(self, path: Path) -> bool:
        path = Path(path)
        config_path = path / 'bert_config.pkl'
        matrix_path = path / 'bert_matrix.npy'
        if not all((p.exists() for p in [config_path, matrix_path])):
            return False
        if not _st_available:
            raise RuntimeError('sentence-transformers is required.')
        with open(config_path, 'rb') as f:
            cfg = pickle.load(f)
        self.doc_ids = cfg['doc_ids']
        self._model = SentenceTransformer(cfg['model_name'])
        self._matrix = np.load(str(matrix_path))
        logger.info(f'BERT index loaded ← {path}  (shape={self._matrix.shape})')
        return True