import logging
import pickle
import threading
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
logger = logging.getLogger('topic_service.lda_model')

def _identity_tokenizer(text: str) -> list[str]:
    return text.split()

class LDAModel:

    def __init__(self):
        self._lda: Optional[LatentDirichletAllocation] = None
        self._vectorizer: Optional[CountVectorizer] = None
        self._doc_ids: list[str] = []
        self._doc_topic_ids: list[int] = []
        self._doc_topic_probs: list[float] = []
        self._topics: list[dict] = []
        self._doc_topic_map: dict[str, dict] = {}
        self._num_topics: int = 0
        self._lock = threading.Lock()

    def build(self, documents: list[dict], num_topics: int=10, num_top_words: int=8) -> None:
        logger.info(f'Training LDA: {len(documents)} docs, {num_topics} topics, {num_top_words} top words')
        self._num_topics = num_topics
        self._doc_ids = [d['doc_id'] for d in documents]
        corpus = [d.get('cleaned', ' '.join(d.get('tokens', []))) for d in documents]
        self._vectorizer = CountVectorizer(tokenizer=_identity_tokenizer, preprocessor=None, lowercase=False, token_pattern=None, max_df=0.95, min_df=5, max_features=20000)
        doc_term_matrix = self._vectorizer.fit_transform(corpus)
        logger.info(f'Doc-term matrix: {doc_term_matrix.shape}')
        self._lda = LatentDirichletAllocation(n_components=num_topics, max_iter=15, learning_method='online', batch_size=512, random_state=42, n_jobs=-1)
        doc_topic_matrix = self._lda.fit_transform(doc_term_matrix)
        logger.info('LDA training complete.')
        self._doc_topic_ids = np.argmax(doc_topic_matrix, axis=1).tolist()
        self._doc_topic_probs = doc_topic_matrix.max(axis=1).tolist()
        feature_names = self._vectorizer.get_feature_names_out()
        self._topics = []
        topic_doc_counts = [0] * num_topics
        for tid in self._doc_topic_ids:
            topic_doc_counts[tid] += 1
        for topic_idx, topic_dist in enumerate(self._lda.components_):
            top_word_indices = topic_dist.argsort()[-num_top_words:][::-1]
            top_words = [feature_names[i] for i in top_word_indices]
            label = ' / '.join(top_words[:3])
            self._topics.append({'topic_id': topic_idx, 'label': label, 'top_words': top_words, 'doc_count': topic_doc_counts[topic_idx]})
        self._doc_topic_map = {}
        for i, doc_id in enumerate(self._doc_ids):
            tid = self._doc_topic_ids[i]
            topic_def = self._topics[tid]
            self._doc_topic_map[doc_id] = {'topic_id': tid, 'topic_label': topic_def['label'], 'top_words': topic_def['top_words'], 'probability': round(float(self._doc_topic_probs[i]), 4)}
        logger.info(f'LDA ready: {num_topics} topics, {len(self._doc_topic_map)} documents mapped.')

    def get_doc_topic(self, doc_id: str) -> Optional[dict]:
        return self._doc_topic_map.get(doc_id)

    def get_doc_topics(self, doc_ids: list[str]) -> list[Optional[dict]]:
        return [self._doc_topic_map.get(did) for did in doc_ids]

    def get_all_topics(self) -> list[dict]:
        return self._topics

    def infer_topic(self, cleaned_text: str) -> dict:
        if self._lda is None or self._vectorizer is None:
            raise RuntimeError('LDA model not built.')
        vec = self._vectorizer.transform([cleaned_text])
        dist = self._lda.transform(vec)[0]
        tid = int(np.argmax(dist))
        prob = float(dist[tid])
        topic_def = self._topics[tid]
        return {'topic_id': tid, 'topic_label': topic_def['label'], 'top_words': topic_def['top_words'], 'probability': round(prob, 4)}

    @property
    def num_topics(self) -> int:
        return self._num_topics

    @property
    def is_built(self) -> bool:
        return self._lda is not None

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'lda': self._lda, 'vectorizer': self._vectorizer, 'doc_ids': self._doc_ids, 'doc_topic_ids': self._doc_topic_ids, 'doc_topic_probs': self._doc_topic_probs, 'topics': self._topics, 'doc_topic_map': self._doc_topic_map, 'num_topics': self._num_topics}, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f'LDA model saved → {path}')

    def load(self, path: Path) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        with open(path, 'rb') as f:
            data = pickle.load(f)
        if 'vec' in data:
            self._lda = data['lda']
            self._vectorizer = data['vec']
            self._topics = data['topics']
            self._doc_topic_map = data['map']
            self._num_topics = data['n']
            self._doc_ids = list(self._doc_topic_map.keys())
            self._doc_topic_ids = [v['topic_id'] for v in self._doc_topic_map.values()]
            self._doc_topic_probs = [v['probability'] for v in self._doc_topic_map.values()]
        else:
            self._lda = data['lda']
            self._vectorizer = data['vectorizer']
            self._doc_ids = data['doc_ids']
            self._doc_topic_ids = data['doc_topic_ids']
            self._doc_topic_probs = data['doc_topic_probs']
            self._topics = data['topics']
            self._doc_topic_map = data['doc_topic_map']
            self._num_topics = data['num_topics']
        logger.info(f'LDA model loaded ← {path} ({self._num_topics} topics, {len(self._doc_topic_map)} docs)')
        return True