import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from preprocessing_service.schemas import TextRequest, BatchTextRequest, PreprocessOptions, TextResponse, BatchTextResponse, DatasetStatusResponse
from preprocessing_service.preprocessor import Preprocessor
from preprocessing_service.dataset_loader import DatasetLoader
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('preprocessing_service')
preprocessor: Preprocessor = None
dataset_loader: DatasetLoader = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global preprocessor, dataset_loader
    logger.info('Starting Preprocessing Service — downloading NLTK data if needed…')
    preprocessor = Preprocessor()
    dataset_loader = DatasetLoader(preprocessor=preprocessor)
    logger.info('Preprocessing Service ready.')
    yield
    logger.info('Preprocessing Service shutting down.')
app = FastAPI(title='Preprocessing Service', description='Handles text normalization, stemming, lemmatization, and dataset loading / caching for the IR system.', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

@app.get('/health', tags=['meta'])
def health():
    return {'status': 'ok', 'service': 'preprocessing'}

@app.post('/preprocess/text', response_model=TextResponse, tags=['preprocessing'])
def preprocess_text(request: TextRequest):
    tokens = preprocessor.preprocess(text=request.text, options=request.options or PreprocessOptions())
    return TextResponse(original=request.text, tokens=tokens, cleaned=' '.join(tokens))

@app.post('/preprocess/batch', response_model=BatchTextResponse, tags=['preprocessing'])
def preprocess_batch(request: BatchTextRequest):
    results: list[TextResponse] = []
    opts = request.options or PreprocessOptions()
    for text in request.texts:
        tokens = preprocessor.preprocess(text=text, options=opts)
        results.append(TextResponse(original=text, tokens=tokens, cleaned=' '.join(tokens)))
    return BatchTextResponse(results=results)

@app.post('/dataset/load', tags=['dataset'])
def load_dataset(dataset_name: str=Query(..., description="e.g. 'quora' or 'msmarco'"), background_tasks: BackgroundTasks=None):
    if dataset_loader.is_loading(dataset_name):
        return {'message': f"Dataset '{dataset_name}' is already being loaded."}
    background_tasks.add_task(dataset_loader.load_dataset, dataset_name)
    return {'message': f"Loading of '{dataset_name}' started in background."}

@app.get('/dataset/status', response_model=DatasetStatusResponse, tags=['dataset'])
def dataset_status(dataset_name: str=Query(..., description="e.g. 'quora' or 'msmarco'")):
    return dataset_loader.get_status(dataset_name)

@app.get('/dataset/docs', tags=['dataset'])
def get_documents(dataset_name: str=Query(...), offset: int=Query(0, ge=0), limit: int=Query(100, ge=1, le=10000)):
    docs = dataset_loader.get_documents(dataset_name, offset=offset, limit=limit)
    if docs is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not loaded. Call POST /dataset/load first.")
    return {'dataset': dataset_name, 'offset': offset, 'count': len(docs), 'documents': docs}

@app.post('/dataset/raw-docs/batch', tags=['dataset'])
def get_raw_docs_batch(dataset_name: str=Query(...), doc_ids: list[str]=Body(...)):
    docs = dataset_loader.get_raw_documents_batch(dataset_name, doc_ids)
    if docs is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not loaded.")
    return {'dataset': dataset_name, 'documents': docs}

@app.get('/dataset/queries', tags=['dataset'])
def get_queries(dataset_name: str=Query(...), offset: int=Query(0, ge=0), limit: int=Query(100, ge=1, le=10000)):
    queries = dataset_loader.get_queries(dataset_name, offset=offset, limit=limit)
    if queries is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not loaded.")
    return {'dataset': dataset_name, 'offset': offset, 'count': len(queries), 'queries': queries}

@app.get('/dataset/qrels', tags=['dataset'])
def get_qrels(dataset_name: str=Query(...)):
    qrels = dataset_loader.get_qrels(dataset_name)
    if qrels is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not loaded.")
    return {'dataset': dataset_name, 'qrels': qrels}

@app.get('/dataset/raw_docs', tags=['dataset'])
def get_raw_documents(dataset_name: str=Query(...), offset: int=Query(0, ge=0), limit: int=Query(100, ge=1, le=10000)):
    docs = dataset_loader.get_raw_documents(dataset_name, offset=offset, limit=limit)
    if docs is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not loaded.")
    return {'dataset': dataset_name, 'offset': offset, 'count': len(docs), 'documents': docs}

@app.get('/dataset/all_doc_ids', tags=['dataset'])
def get_all_doc_ids(dataset_name: str=Query(...)):
    doc_ids = dataset_loader.get_all_doc_ids(dataset_name)
    if doc_ids is None:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not loaded.")
    return {'dataset': dataset_name, 'count': len(doc_ids), 'doc_ids': doc_ids}