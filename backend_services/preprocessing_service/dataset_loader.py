import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional
import ir_datasets
from preprocessing_service.preprocessor import Preprocessor
from preprocessing_service.schemas import DatasetStatusResponse, PreprocessOptions
logger = logging.getLogger('preprocessing_service.dataset_loader')
_BASE_DIR = Path(__file__).resolve().parent.parent.parent / 'data'
DATASET_REGISTRY: dict[str, str] = {'quora': 'beir/quora/test', 'msmarco': 'beir/msmarco/test'}
MSMARCO_MAX_DOCS = 200000
DEFAULT_PREPROCESS_OPTS = PreprocessOptions(lowercase=True, remove_punctuation=True, remove_stopwords=True, stem=True, lemmatize=False)

class _DatasetState:

    def __init__(self, name: str):
        self.name = name
        self.status: str = 'not_loaded'
        self.error: Optional[str] = None
        self.progress_docs: int = 0
        self.total_docs: int = 0
        self.progress_queries: int = 0
        self.total_queries: int = 0
        self.raw_docs: dict[str, str] = {}
        self.preprocessed_docs: list[dict] = []
        self.doc_ids: list[str] = []
        self.queries: list[dict] = []
        self.qrels: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    def to_status_response(self) -> DatasetStatusResponse:
        with self._lock:
            return DatasetStatusResponse(dataset=self.name, status=self.status, progress_docs=self.progress_docs, total_docs=self.total_docs, progress_queries=self.progress_queries, total_queries=self.total_queries, error=self.error)

class DatasetLoader:

    def __init__(self, preprocessor: Preprocessor):
        self._preprocessor = preprocessor
        self._states: dict[str, _DatasetState] = {}
        self._states_lock = threading.Lock()

    def is_loading(self, dataset_name: str) -> bool:
        with self._states_lock:
            s = self._states.get(dataset_name)
        return s is not None and s.status == 'loading'

    def get_status(self, dataset_name: str) -> DatasetStatusResponse:
        with self._states_lock:
            s = self._states.get(dataset_name)
        if s is None:
            return DatasetStatusResponse(dataset=dataset_name, status='not_loaded')
        return s.to_status_response()

    def get_documents(self, dataset_name: str, offset: int=0, limit: int=100) -> Optional[list[dict]]:
        s = self._get_ready_state(dataset_name)
        if s is None:
            return None
        return s.preprocessed_docs[offset:offset + limit]

    def get_raw_documents(self, dataset_name: str, offset: int=0, limit: int=100) -> Optional[list[dict]]:
        s = self._get_ready_state(dataset_name)
        if s is None:
            return None
        subset_ids = s.doc_ids[offset:offset + limit]
        return [{'doc_id': did, 'text': s.raw_docs.get(did, '')} for did in subset_ids]

    def get_raw_documents_batch(self, dataset_name: str, doc_ids: list[str]) -> Optional[list[dict]]:
        s = self._get_ready_state(dataset_name)
        if s is None:
            return None
        return [{'doc_id': did, 'text': s.raw_docs.get(did, '')} for did in doc_ids]

    def get_queries(self, dataset_name: str, offset: int=0, limit: int=100) -> Optional[list[dict]]:
        s = self._get_ready_state(dataset_name)
        if s is None:
            return None
        return s.queries[offset:offset + limit]

    def get_qrels(self, dataset_name: str) -> Optional[dict]:
        s = self._get_ready_state(dataset_name)
        if s is None:
            return None
        return s.qrels

    def get_all_doc_ids(self, dataset_name: str) -> Optional[list[str]]:
        s = self._get_ready_state(dataset_name)
        if s is None:
            return None
        return s.doc_ids

    def load_dataset(self, dataset_name: str) -> None:
        state = self._get_or_create_state(dataset_name)
        with state._lock:
            state.status = 'loading'
            state.error = None
        try:
            cache_dir = _BASE_DIR / dataset_name
            cache_dir.mkdir(parents=True, exist_ok=True)
            if self._try_load_from_cache(state, cache_dir):
                logger.info(f'[{dataset_name}] Loaded from disk cache.')
                with state._lock:
                    state.status = 'ready'
                return
            ir_key = DATASET_REGISTRY.get(dataset_name)
            if ir_key is None:
                raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(DATASET_REGISTRY.keys())}")
            logger.info(f'[{dataset_name}] Loading from ir_datasets: {ir_key}')
            dataset = ir_datasets.load(ir_key)
            self._load_documents(state, dataset, dataset_name, cache_dir)
            self._load_queries(state, dataset, cache_dir)
            self._load_qrels(state, dataset, cache_dir)
            with state._lock:
                state.status = 'ready'
            logger.info(f'[{dataset_name}] Dataset fully loaded and cached.')
        except Exception as exc:
            logger.exception(f'[{dataset_name}] Load failed: {exc}')
            with state._lock:
                state.status = 'error'
                state.error = str(exc)

    def _get_or_create_state(self, dataset_name: str) -> _DatasetState:
        with self._states_lock:
            if dataset_name not in self._states:
                self._states[dataset_name] = _DatasetState(dataset_name)
            return self._states[dataset_name]

    def _get_ready_state(self, dataset_name: str) -> Optional[_DatasetState]:
        with self._states_lock:
            s = self._states.get(dataset_name)
        if s is None or s.status != 'ready':
            return None
        return s

    def _load_documents(self, state: _DatasetState, dataset, dataset_name: str, cache_dir: Path) -> None:
        max_docs = MSMARCO_MAX_DOCS if dataset_name == 'msmarco' else None
        preprocessed: list[dict] = []
        raw_docs: dict[str, str] = {}
        doc_ids: list[str] = []
        opts = DEFAULT_PREPROCESS_OPTS
        logger.info(f'[{dataset_name}] Preprocessing documents…')
        t0 = time.time()
        count = 0
        for doc in dataset.docs_iter():
            doc_id = str(doc.doc_id)
            raw_text = getattr(doc, 'text', '') or getattr(doc, 'body', '') or ''
            tokens = self._preprocessor.preprocess(raw_text, options=opts)
            preprocessed.append({'doc_id': doc_id, 'tokens': tokens, 'cleaned': ' '.join(tokens)})
            raw_docs[doc_id] = raw_text
            doc_ids.append(doc_id)
            count += 1
            if count % 5000 == 0:
                with state._lock:
                    state.progress_docs = count
                logger.debug(f'[{dataset_name}] Processed {count} docs…')
            if max_docs and count >= max_docs:
                logger.info(f'[{dataset_name}] Reached max_docs limit ({max_docs}). Stopping.')
                break
        elapsed = time.time() - t0
        logger.info(f'[{dataset_name}] Preprocessed {count} docs in {elapsed:.1f}s ({count / max(elapsed, 1):.0f} docs/s)')
        self._save_jsonl(cache_dir / 'preprocessed_docs.jsonl', preprocessed)
        with open(cache_dir / 'raw_docs.json', 'w', encoding='utf-8') as f:
            json.dump(raw_docs, f, ensure_ascii=False)
        with open(cache_dir / 'doc_ids.json', 'w', encoding='utf-8') as f:
            json.dump(doc_ids, f)
        with state._lock:
            state.preprocessed_docs = preprocessed
            state.raw_docs = raw_docs
            state.doc_ids = doc_ids
            state.total_docs = count
            state.progress_docs = count

    def _load_queries(self, state: _DatasetState, dataset, cache_dir: Path) -> None:
        opts = DEFAULT_PREPROCESS_OPTS
        queries: list[dict] = []
        count = 0
        for query in dataset.queries_iter():
            query_id = str(query.query_id)
            text = query.text
            tokens = self._preprocessor.preprocess(text, options=opts)
            queries.append({'query_id': query_id, 'original': text, 'tokens': tokens, 'cleaned': ' '.join(tokens)})
            count += 1
        logger.info(f'Loaded {count} queries.')
        self._save_jsonl(cache_dir / 'queries.jsonl', queries)
        with state._lock:
            state.queries = queries
            state.total_queries = count
            state.progress_queries = count

    def _load_qrels(self, state: _DatasetState, dataset, cache_dir: Path) -> None:
        qrels: dict[str, dict[str, int]] = {}
        count = 0
        for qrel in dataset.qrels_iter():
            qid = str(qrel.query_id)
            did = str(qrel.doc_id)
            rel = int(qrel.relevance)
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][did] = rel
            count += 1
        logger.info(f'Loaded {count} qrel entries.')
        with open(cache_dir / 'qrels.json', 'w', encoding='utf-8') as f:
            json.dump(qrels, f)
        with state._lock:
            state.qrels = qrels

    def _try_load_from_cache(self, state: _DatasetState, cache_dir: Path) -> bool:
        docs_path = cache_dir / 'preprocessed_docs.jsonl'
        raw_path = cache_dir / 'raw_docs.json'
        doc_ids_path = cache_dir / 'doc_ids.json'
        queries_path = cache_dir / 'queries.jsonl'
        qrels_path = cache_dir / 'qrels.json'
        required = [docs_path, raw_path, doc_ids_path, queries_path, qrels_path]
        if not all((p.exists() for p in required)):
            return False
        logger.info(f'[{state.name}] Reading from disk cache…')
        preprocessed = self._load_jsonl(docs_path)
        with open(raw_path, 'r', encoding='utf-8') as f:
            raw_docs = json.load(f)
        with open(doc_ids_path, 'r', encoding='utf-8') as f:
            doc_ids = json.load(f)
        queries = self._load_jsonl(queries_path)
        with open(qrels_path, 'r', encoding='utf-8') as f:
            qrels = json.load(f)
        with state._lock:
            state.preprocessed_docs = preprocessed
            state.raw_docs = raw_docs
            state.doc_ids = doc_ids
            state.queries = queries
            state.qrels = qrels
            state.total_docs = len(preprocessed)
            state.progress_docs = len(preprocessed)
            state.total_queries = len(queries)
            state.progress_queries = len(queries)
        return True

    @staticmethod
    def _save_jsonl(path: Path, records: list[dict]) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict]:
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records