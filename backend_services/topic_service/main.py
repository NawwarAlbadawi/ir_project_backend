import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from topic_service.lda_model import LDAModel
from topic_service.schemas import BuildTopicsRequest, BuildTopicsResponse, DocTopicRequest, DocTopicResponse, DocTopicResult, TopicDefinition, TopicStatusResponse
logger = logging.getLogger('topic_service')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
PREPROCESSING_URL = 'http://localhost:8001'
DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'
PAGE_SIZE = 5000

class _TopicState:

    def __init__(self, name: str):
        self.name = name
        self.status: str = 'not_built'
        self.error: str | None = None
        self.model = LDAModel()
        self._lock = threading.Lock()
_states: dict[str, _TopicState] = {}
_states_lock = threading.Lock()

def _get_or_create(dataset_name: str) -> _TopicState:
    with _states_lock:
        if dataset_name not in _states:
            _states[dataset_name] = _TopicState(dataset_name)
        return _states[dataset_name]

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info('Topic Service starting — scanning for cached LDA models…')
    if DATA_DIR.exists():
        for dataset_dir in DATA_DIR.iterdir():
            if dataset_dir.is_dir():
                state = _get_or_create(dataset_dir.name)
                pkl_path = dataset_dir / 'lda_model.pkl'
                if state.model.load(pkl_path):
                    with state._lock:
                        state.status = 'ready'
                    logger.info(f'[{dataset_dir.name}] LDA model restored from cache.')
    logger.info('Topic Service ready.')
    yield
    logger.info('Topic Service shutting down.')
app = FastAPI(title='Topic Detection Service', description='LDA-based topic modeling. Discovers latent topics in the corpus and assigns each document a topic.', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

@app.get('/health', tags=['meta'])
def health():
    return {'status': 'ok', 'service': 'topic_detection'}

@app.post('/topics/build', response_model=BuildTopicsResponse, tags=['topics'])
def build_topics(request: BuildTopicsRequest, background_tasks: BackgroundTasks):
    state = _get_or_create(request.dataset_name)
    if state.status == 'building':
        raise HTTPException(409, f"Already building topics for '{request.dataset_name}'.")
    background_tasks.add_task(_build_pipeline, state, request)
    return BuildTopicsResponse(dataset=request.dataset_name, status='building', num_topics=request.num_topics)

@app.get('/topics/status', response_model=TopicStatusResponse, tags=['topics'])
def topic_status(dataset_name: str=Query(...)):
    state = _get_or_create(dataset_name)
    with state._lock:
        return TopicStatusResponse(dataset=dataset_name, status=state.status, num_topics=state.model.num_topics, error=state.error)

@app.get('/topics/all', tags=['topics'])
def get_all_topics(dataset_name: str=Query(...)):
    state = _get_or_create(dataset_name)
    if state.status != 'ready':
        raise HTTPException(404, f"Topic model not ready for '{dataset_name}'.")
    topics = state.model.get_all_topics()
    return {'dataset': dataset_name, 'num_topics': state.model.num_topics, 'topics': topics}

@app.post('/topics/detect', response_model=DocTopicResponse, tags=['topics'])
def detect_topics(request: DocTopicRequest):
    state = _get_or_create(request.dataset_name)
    if state.status != 'ready':
        raise HTTPException(404, f"Topic model not built for '{request.dataset_name}'. Call POST /topics/build first.")
    results: list[DocTopicResult] = []
    for doc_id in request.doc_ids:
        info = state.model.get_doc_topic(doc_id)
        if info:
            results.append(DocTopicResult(doc_id=doc_id, topic_id=info['topic_id'], topic_label=info['topic_label'], top_words=info['top_words'], probability=info['probability']))
        else:
            results.append(DocTopicResult(doc_id=doc_id, topic_id=-1, topic_label='unknown', top_words=[], probability=0.0))
    return DocTopicResponse(dataset=request.dataset_name, results=results)

@app.get('/topics/infer', tags=['topics'])
def infer_query_topic(dataset_name: str=Query(...), text: str=Query(..., description='Preprocessed query text to infer topic for')):
    state = _get_or_create(dataset_name)
    if state.status != 'ready':
        raise HTTPException(404, f"Topic model not ready for '{dataset_name}'.")
    try:
        topic = state.model.infer_topic(text)
        return {'dataset': dataset_name, 'input': text, 'topic': topic}
    except Exception as e:
        raise HTTPException(500, str(e))

def _build_pipeline(state: _TopicState, request: BuildTopicsRequest) -> None:
    with state._lock:
        state.status = 'building'
        state.error = None
    try:
        docs = _fetch_all_documents(request.dataset_name)
        state.model.build(documents=docs, num_topics=request.num_topics, num_top_words=request.num_top_words)
        cache_dir = DATA_DIR / request.dataset_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        state.model.save(cache_dir / 'lda_model.pkl')
        with state._lock:
            state.status = 'ready'
        logger.info(f'[{request.dataset_name}] LDA topic model built and saved.')
    except Exception as exc:
        logger.exception(f'[{request.dataset_name}] Topic build failed: {exc}')
        with state._lock:
            state.status = 'error'
            state.error = str(exc)

def _fetch_all_documents(dataset_name: str) -> list[dict]:
    docs: list[dict] = []
    offset = 0
    logger.info(f'[{dataset_name}] Fetching documents from Preprocessing Service…')
    with httpx.Client(timeout=120.0) as client:
        while True:
            resp = client.get(f'{PREPROCESSING_URL}/dataset/docs', params={'dataset_name': dataset_name, 'offset': offset, 'limit': PAGE_SIZE})
            resp.raise_for_status()
            batch = resp.json().get('documents', [])
            if not batch:
                break
            docs.extend(batch)
            offset += len(batch)
            logger.debug(f'[{dataset_name}] Fetched {len(docs)} docs…')
    logger.info(f'[{dataset_name}] Total docs fetched: {len(docs)}')
    return docs