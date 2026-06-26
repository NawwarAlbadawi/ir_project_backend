import json

TOKEN = "3EqzK9BvXDwGI3rgFQ9jMvnSFY7_6Nvp6CF6e8oAmK64Kc9Bp"
DRIVE_FILE_ID = "1ZLIdrsobW4nA6Lu02h5ZdY5NwO6OjPdD"

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
        "gpuClass": "standard"
    },
    "cells": []
}

def code_cell(cid, lines):
    src = "\n".join(lines) if isinstance(lines, list) else lines
    return {"cell_type": "code", "execution_count": None, "id": cid,
            "metadata": {}, "outputs": [], "source": src}

def md_cell(cid, src):
    return {"cell_type": "markdown", "id": cid, "metadata": {}, "source": src}

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(md_cell("md-title", (
    "# IR Project — Complete Google Colab Notebook\n\n"
    "> **Before running:** Enable GPU → Runtime → Change runtime type → **T4 GPU**\n\n"
    "| Cell | Purpose |\n|---|---|\n"
    "| 1 | Download & Extract project |\n"
    "| 2 | Install all libraries |\n"
    "| 3 | Set Python path & start all 7 services |\n"
    "| 4 | NLTK data |\n"
    "| 5 | ngrok public URL (optional) |\n"
    "| 6 | Health check |\n"
    "| 7 | Load Quora dataset |\n"
    "| 8 | Build search indexes |\n"
    "| 9 | Build LDA topic model ⭐ |\n"
    "| 10 | View all discovered topics ⭐ |\n"
    "| 11 | Search with topic labels ⭐ |\n"
    "| 12 | Regular search |\n"
    "| 13 | Compare all 6 models |\n"
    "| 14 | Evaluate (MAP, nDCG) |\n"
    "| 15 | View logs (debug) |\n"
    "| 16 | Save all indexes + LDA to Google Drive |\n"
    "| 16b | Restore from Drive (next session fast start) |"
)))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Download & Extract
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-01", [
    "# CELL 1 — Download & Extract IR Project from Google Drive",
    "!pip install -q gdown",
    "import gdown, zipfile, os, shutil",
    "",
    f"FILE_ID   = '{DRIVE_FILE_ID}'",
    "ZIP_PATH  = '/content/IR_Project.zip'",
    "PROJ_PATH = '/content/IR_Project'",
    "",
    "# Clear old extraction if any",
    "if os.path.exists(PROJ_PATH): shutil.rmtree(PROJ_PATH)",
    "",
    "print('Downloading...')",
    "gdown.download(id=FILE_ID, output=ZIP_PATH, quiet=False)",
    "",
    "print('Extracting...')",
    "with zipfile.ZipFile(ZIP_PATH, 'r') as z:",
    "    z.extractall(PROJ_PATH)",
    "",
    "# Auto-detect backend path",
    "BACKEND_PATH = None",
    "for root, dirs, files in os.walk(PROJ_PATH):",
    "    if 'backend_services' in dirs:",
    "        BACKEND_PATH = os.path.join(root, 'backend_services')",
    "        break",
    "if not BACKEND_PATH:",
    "    BACKEND_PATH = PROJ_PATH + '/IR_Project/backend_services'",
    "",
    "print(f'Backend: {BACKEND_PATH}')",
    "print(f'Services: {[d for d in os.listdir(BACKEND_PATH) if not d.startswith(\"_\")]}')",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Install libraries
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-02", [
    "# CELL 2 — Install All Libraries (~2-3 min)",
    "import subprocess",
    "subprocess.run(['pip', 'install', '-q',",
    "    'fastapi>=0.111.0', 'uvicorn[standard]>=0.29.0', 'httpx>=0.27.0',",
    "    'pydantic>=2.0.0', 'nltk>=3.8.1', 'pyspellchecker>=0.8.1',",
    "    'ir-datasets>=0.5.5', 'scikit-learn>=1.4.0', 'rank-bm25>=0.2.2',",
    "    'scipy>=1.13.0', 'numpy>=1.26.0', 'gensim>=4.3.0',",
    "    'sentence-transformers>=2.7.0', 'pyngrok'])",
    "print('All packages installed!')",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Set path + create topic_service + start ALL 7 services
# ─────────────────────────────────────────────────────────────────────────────
cell3 = [
    "# CELL 3 — Set Path + Create topic_service + Start ALL 7 Services",
    "import os, sys, subprocess, time",
    "",
    "# Auto-detect BACKEND_PATH",
    "PROJ_PATH = '/content/IR_Project'",
    "BACKEND_PATH = None",
    "for root, dirs, files in os.walk(PROJ_PATH):",
    "    if 'backend_services' in dirs:",
    "        BACKEND_PATH = os.path.join(root, 'backend_services')",
    "        break",
    "if not BACKEND_PATH:",
    "    BACKEND_PATH = '/content/IR_Project/IR_Project/backend_services'",
    "if BACKEND_PATH not in sys.path:",
    "    sys.path.insert(0, BACKEND_PATH)",
    "print(f'Backend: {BACKEND_PATH}')",
    "",
    "# Fix data directories",
    "data_base = os.path.normpath(os.path.join(BACKEND_PATH, '..', 'data'))",
    "os.makedirs(os.path.join(data_base, 'quora', 'word2vec'), exist_ok=True)",
    "os.makedirs(os.path.join(data_base, 'quora', 'bert'), exist_ok=True)",
    "",
    "# Create topic_service (not in zip)",
    "topic_dir = os.path.join(BACKEND_PATH, 'topic_service')",
    "os.makedirs(topic_dir, exist_ok=True)",
    "",
    "open(os.path.join(topic_dir, '__init__.py'), 'w').write('\"\"\"Topic Detection Service.\"\"\"\\n')",
    "",
    "open(os.path.join(topic_dir, 'schemas.py'), 'w').write(",
    "'from pydantic import BaseModel, Field\\nfrom typing import Optional\\n'",
    "'class BuildTopicsRequest(BaseModel):\\n    dataset_name: str\\n    num_topics: int = Field(10, ge=2, le=50)\\n    num_top_words: int = Field(8, ge=3, le=20)\\n'",
    "'class TopicDefinition(BaseModel):\\n    topic_id: int; label: str; top_words: list[str]; doc_count: int\\n'",
    "'class BuildTopicsResponse(BaseModel):\\n    dataset: str; status: str; num_topics: int\\n    topics: list[TopicDefinition] = []; error: Optional[str] = None\\n'",
    "'class DocTopicRequest(BaseModel):\\n    doc_ids: list[str]; dataset_name: str\\n'",
    "'class DocTopicResult(BaseModel):\\n    doc_id: str; topic_id: int; topic_label: str; top_words: list[str]; probability: float\\n'",
    "'class DocTopicResponse(BaseModel):\\n    dataset: str; results: list[DocTopicResult]\\n'",
    "'class TopicStatusResponse(BaseModel):\\n    dataset: str; status: str; num_topics: int = 0; error: Optional[str] = None\\n')",
    "",
    "open(os.path.join(topic_dir, 'lda_model.py'), 'w').write(",
    "'import logging, pickle\\nfrom pathlib import Path\\nimport numpy as np\\n'",
    "'from sklearn.decomposition import LatentDirichletAllocation\\nfrom sklearn.feature_extraction.text import CountVectorizer\\n'",
    "'logger = logging.getLogger(\"topic_service.lda\")\\n'",
    "'class LDAModel:\\n'",
    "'    def __init__(self): self._lda=self._vec=None;self._topics=[];self._map={};self._n=0\\n'",
    "'    def get_doc_topic(self,doc_id): return self._map.get(doc_id)\\n'",
    "'    def get_all_topics(self): return self._topics\\n'",
    "'    def infer_topic(self,text):\\n'",
    "'        v=self._vec.transform([text]);d=self._lda.transform(v)[0];i=int(np.argmax(d));t=self._topics[i]\\n'",
    "'        return{\"topic_id\":i,\"topic_label\":t[\"label\"],\"top_words\":t[\"top_words\"],\"probability\":round(float(d[i]),4)}\\n'",
    "'    @property\\n    def num_topics(self): return self._n\\n'",
    "'    @property\\n    def is_built(self): return self._lda is not None\\n'",
    "'    def save(self,path):\\n'",
    "'        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)\\n'",
    "'        with open(p,\"wb\") as f: pickle.dump({\"lda\":self._lda,\"vec\":self._vec,\"topics\":self._topics,\"map\":self._map,\"n\":self._n},f)\\n'",
    "'    def load(self,path):\\n'",
    "'        p=Path(path)\\n'",
    "'        if not p.exists(): return False\\n'",
    "'        with open(p,\"rb\") as f: data=pickle.load(f)\\n'",
    "'        self._lda=data[\"lda\"];self._vec=data[\"vec\"];self._topics=data[\"topics\"];self._map=data[\"map\"];self._n=data[\"n\"]\\n'",
    "'        return True\\n')",
    "",
    "open(os.path.join(topic_dir, 'main.py'), 'w').write(",
    "'import logging,threading\\nfrom contextlib import asynccontextmanager\\nfrom pathlib import Path\\nimport httpx\\n'",
    "'from fastapi import FastAPI,HTTPException,BackgroundTasks,Query\\nfrom fastapi.middleware.cors import CORSMiddleware\\n'",
    "'from topic_service.lda_model import LDAModel\\n'",
    "'from topic_service.schemas import (BuildTopicsRequest,BuildTopicsResponse,DocTopicRequest,DocTopicResponse,DocTopicResult,TopicStatusResponse)\\n'",
    "'logger=logging.getLogger(\"topic_service\")\\nlogging.basicConfig(level=logging.INFO)\\n'",
    "'DATA=Path(__file__).resolve().parent.parent.parent/\"data\"\\n'",
    "'class State:\\n    def __init__(self,n): self.name=n;self.status=\"not_built\";self.error=None;self.model=LDAModel();self._lock=threading.Lock()\\n'",
    "'_states={};_lock=threading.Lock()\\n'",
    "'def _get(n):\\n    with _lock:\\n        if n not in _states:_states[n]=State(n)\\n        return _states[n]\\n'",
    "'@asynccontextmanager\\nasync def lifespan(app):\\n'",
    "'    if DATA.exists():\\n        for d in DATA.iterdir():\\n            if d.is_dir():\\n                s=_get(d.name)\\n                if s.model.load(d/\"lda_model.pkl\"):\\n                    with s._lock:s.status=\"ready\"\\n'",
    "'    logger.info(\"Topic Service ready.\");yield\\n'",
    "'app=FastAPI(title=\"Topic Detection Service\",lifespan=lifespan)\\n'",
    "'app.add_middleware(CORSMiddleware,allow_origins=[\"*\"],allow_methods=[\"*\"],allow_headers=[\"*\"])\\n'",
    "'@app.get(\"/health\")\\ndef health():return{\"status\":\"ok\",\"service\":\"topic_detection\"}\\n'",
    "'@app.get(\"/topics/status\",response_model=TopicStatusResponse)\\n'",
    "'def tstatus(dataset_name:str=Query(...)):\\n    s=_get(dataset_name)\\n    with s._lock:return TopicStatusResponse(dataset=dataset_name,status=s.status,num_topics=s.model.num_topics,error=s.error)\\n'",
    "'@app.get(\"/topics/all\")\\n'",
    "'def all_t(dataset_name:str=Query(...)):\\n    s=_get(dataset_name)\\n    if s.status!=\"ready\":raise HTTPException(404,\"Not ready.\")\\n    return{\"dataset\":dataset_name,\"num_topics\":s.model.num_topics,\"topics\":s.model.get_all_topics()}\\n'",
    "'@app.post(\"/topics/detect\",response_model=DocTopicResponse)\\n'",
    "'def detect(req:DocTopicRequest):\\n    s=_get(req.dataset_name)\\n    if s.status!=\"ready\":raise HTTPException(404,\"Build topics first.\")\\n    res=[]\\n'",
    "'    for did in req.doc_ids:\\n        info=s.model.get_doc_topic(did)\\n'",
    "'        if info:res.append(DocTopicResult(doc_id=did,topic_id=info[\"topic_id\"],topic_label=info[\"topic_label\"],top_words=info[\"top_words\"],probability=info[\"probability\"]))\\n'",
    "'        else:res.append(DocTopicResult(doc_id=did,topic_id=-1,topic_label=\"unknown\",top_words=[],probability=0.0))\\n'",
    "'    return DocTopicResponse(dataset=req.dataset_name,results=res)\\n'",
    "'@app.get(\"/topics/infer\")\\n'",
    "'def infer(dataset_name:str=Query(...),text:str=Query(...)):\\n    s=_get(dataset_name)\\n    if s.status!=\"ready\":raise HTTPException(404,\"Not ready.\")\\n    return{\"dataset\":dataset_name,\"topic\":s.model.infer_topic(text)}\\n')",
    "",
    "print('topic_service files created')",
    "",
    "# IMPORTANT: Set PYTHONPATH so subprocesses can find modules",
    "env = os.environ.copy()",
    "env['PYTHONPATH'] = BACKEND_PATH",
    "",
    "# Kill any old processes",
    "os.system('fuser -k 8000/tcp 8001/tcp 8002/tcp 8003/tcp 8004/tcp 8005/tcp 8006/tcp 2>/dev/null')",
    "time.sleep(3)",
    "",
    "services = [",
    "    ('Preprocessing', 8001, 'preprocessing_service.main:app'),",
    "    ('Indexing',      8002, 'indexing_service.main:app'),",
    "    ('Retrieval',     8003, 'retrieval_service.main:app'),",
    "    ('Evaluation',    8004, 'ranking_eval_service.main:app'),",
    "    ('Refinement',    8005, 'query_refinement_service.main:app'),",
    "    ('Topic',         8006, 'topic_service.main:app'),",
    "    ('API Gateway',   8000, 'api_gateway.main:app'),",
    "]",
    "for name, port, module in services:",
    "    log = open(f'/content/log_{name}.txt', 'w')",
    "    subprocess.Popen(",
    "        ['uvicorn', module, '--host', '0.0.0.0', '--port', str(port), '--log-level', 'warning'],",
    "        cwd=BACKEND_PATH, stdout=log, stderr=log, env=env)",
    "    print(f'  Started {name:20} port={port}')",
    "    time.sleep(3)",
    "",
    "print('Waiting 20s for all services...')",
    "time.sleep(20)",
    "print('Done! Run Cell 4 next.')",
]
nb["cells"].append(code_cell("cell-03", cell3))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — NLTK
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-04", [
    "# CELL 4 — NLTK Data",
    "import nltk",
    "for pkg in ['punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng', 'omw-1.4']:",
    "    nltk.download(pkg, quiet=True)",
    "    print(f'  OK: {pkg}')",
    "print('NLTK ready!')",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — ngrok
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-05", [
    "# CELL 5 — ngrok Public URL (optional — skip if not needed)",
    "from pyngrok import ngrok",
    "",
    f"NGROK_TOKEN = '{TOKEN}'",
    "ngrok.kill()",
    "ngrok.set_auth_token(NGROK_TOKEN)",
    "tunnel = ngrok.connect(8000)",
    "PUBLIC_URL = tunnel.public_url",
    "print('=' * 55)",
    "print(f'  API URL  : {PUBLIC_URL}')",
    "print(f'  API Docs : {PUBLIC_URL}/docs')",
    "print('=' * 55)",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Health Check
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-06", [
    "# CELL 6 — Health Check (All 7 Services)",
    "import httpx",
    "",
    "checks = {",
    "    'Preprocessing  (8001)': 'http://localhost:8001/health',",
    "    'Indexing       (8002)': 'http://localhost:8002/health',",
    "    'Retrieval      (8003)': 'http://localhost:8003/health',",
    "    'Evaluation     (8004)': 'http://localhost:8004/health',",
    "    'Refinement     (8005)': 'http://localhost:8005/health',",
    "    'Topic Detect   (8006)': 'http://localhost:8006/health',",
    "    'API Gateway    (8000)': 'http://localhost:8000/health',",
    "}",
    "",
    "print('Health Check:')",
    "all_ok = True",
    "for name, url in checks.items():",
    "    try:",
    "        r = httpx.get(url, timeout=5)",
    "        ok = r.status_code == 200",
    "        icon = 'OK  ' if ok else 'WARN'",
    "        print(f'  {icon}  {name}')",
    "        if not ok: all_ok = False",
    "    except:",
    "        print(f'  DOWN  {name}')",
    "        all_ok = False",
    "",
    "print()",
    "if all_ok: print('All 7 services healthy!')",
    "else: print('Some not ready. Wait 20s and re-run this cell.')",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Load Dataset
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-07", [
    "# CELL 7 — Load Quora Dataset (10-30 min first time)",
    "import httpx, time",
    "",
    "DATASET = 'quora'",
    "r = httpx.post('http://localhost:8000/dataset/load',",
    "               json={'dataset_name': DATASET}, timeout=30)",
    "print('Response:', r.json())",
    "",
    "while True:",
    "    r = httpx.get(f'http://localhost:8000/dataset/status?dataset_name={DATASET}')",
    "    d = r.json()",
    "    status = d.get('status')",
    "    docs = d.get('progress_docs', 0)",
    "    total = d.get('total_docs', '?')",
    "    print(f'  Status: {status:12}  Docs: {docs}/{total}')",
    "    if status == 'ready':",
    "        print(f'Dataset ready! ({total} docs)')",
    "        break",
    "    elif status == 'error':",
    "        print('Error:', d.get('error'))",
    "        break",
    "    time.sleep(20)",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Build Indexes
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-08", [
    "# CELL 8 — Build Search Indexes (30-60 min, GPU recommended)",
    "import httpx, time",
    "",
    "r = httpx.post('http://localhost:8000/index/build',",
    "    json={'dataset_name': 'quora', 'models': ['tfidf', 'bm25', 'word2vec', 'bert']}, timeout=30)",
    "print('Response:', r.json())",
    "",
    "while True:",
    "    r = httpx.get('http://localhost:8000/index/status?dataset_name=quora')",
    "    d = r.json()",
    "    status = d.get('status')",
    "    built = d.get('built_models', [])",
    "    print(f'  Status: {status:12}  Built: {built}')",
    "    if status == 'ready':",
    "        print('All indexes ready:', built)",
    "        break",
    "    elif status == 'error':",
    "        print('Error:', d.get('error'))",
    "        break",
    "    time.sleep(20)",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Build LDA (directly via ir_datasets, n_jobs=1 to avoid disk issues)
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-09", [
    "# CELL 9 — Build LDA Topic Model ⭐ NEW FEATURE (~5-10 min)",
    "import os, pickle, numpy as np",
    "from pathlib import Path",
    "from sklearn.decomposition import LatentDirichletAllocation",
    "from sklearn.feature_extraction.text import CountVectorizer",
    "import ir_datasets",
    "",
    "print('Loading Quora corpus from ir_datasets...')",
    "dataset = ir_datasets.load('beir/quora/test')",
    "",
    "docs, doc_ids = [], []",
    "for doc in dataset.docs_iter():",
    "    text = (doc.text or '').strip()",
    "    if text:",
    "        docs.append(text)",
    "        doc_ids.append(doc.doc_id)",
    "    if len(docs) % 100000 == 0 and len(docs) > 0:",
    "        print(f'  Loaded {len(docs)} docs...')",
    "",
    "print(f'Total: {len(docs)} documents')",
    "",
    "# Free temp files before training",
    "os.system('rm -rf /tmp/joblib* 2>/dev/null')",
    "",
    "print('Training LDA (10 topics)... this takes ~5 min')",
    "vec = CountVectorizer(max_df=0.95, min_df=5, max_features=15_000, stop_words='english')",
    "dtm = vec.fit_transform(docs)",
    "",
    "lda = LatentDirichletAllocation(",
    "    n_components=10, max_iter=15,",
    "    learning_method='online', batch_size=512,",
    "    random_state=42,",
    "    n_jobs=1)  # n_jobs=1 avoids disk-full errors on Colab",
    "",
    "mat = lda.fit_transform(dtm)",
    "print('Training done!')",
    "",
    "# Build topic map",
    "fn = vec.get_feature_names_out()",
    "tids = np.argmax(mat, axis=1).tolist()",
    "probs = mat.max(axis=1).tolist()",
    "counts = [0] * 10",
    "for t in tids: counts[t] += 1",
    "",
    "topics = []",
    "for i, dist in enumerate(lda.components_):",
    "    idx = dist.argsort()[-8:][::-1]",
    "    words = [fn[j] for j in idx]",
    "    topics.append({'topic_id': i, 'label': '/'.join(words[:3]),",
    "                   'top_words': words, 'doc_count': counts[i]})",
    "",
    "doc_map = {}",
    "for i, did in enumerate(doc_ids):",
    "    t = topics[tids[i]]",
    "    doc_map[did] = {'topic_id': t['topic_id'], 'topic_label': t['label'],",
    "                    'top_words': t['top_words'], 'probability': round(float(probs[i]), 4)}",
    "",
    "print('\\n10 Topics:')",
    "for t in topics:",
    "    print(f\"  [{t['topic_id']}] {t['label']:30} ({t['doc_count']} docs)\")",
    "",
    "# Save model to disk",
    "save_dir = Path(BACKEND_PATH).parent / 'data' / 'quora'",
    "save_dir.mkdir(parents=True, exist_ok=True)",
    "save_path = save_dir / 'lda_model.pkl'",
    "with open(save_path, 'wb') as f:",
    "    pickle.dump({'lda': lda, 'vec': vec, 'topics': topics, 'map': doc_map, 'n': 10}, f)",
    "print(f'\\nModel saved! ({save_path.stat().st_size // 1024 // 1024} MB)')",
    "",
    "# Reload topic service to pick up the saved model",
    "import subprocess, time, httpx",
    "os.system('fuser -k 8006/tcp 2>/dev/null')",
    "time.sleep(2)",
    "env = os.environ.copy()",
    "env['PYTHONPATH'] = BACKEND_PATH",
    "log = open('/content/log_Topic.txt', 'w')",
    "subprocess.Popen(['uvicorn', 'topic_service.main:app', '--host', '0.0.0.0',",
    "                  '--port', '8006', '--log-level', 'warning'],",
    "                 cwd=BACKEND_PATH, stdout=log, stderr=log, env=env)",
    "time.sleep(8)",
    "r = httpx.get('http://localhost:8006/topics/status?dataset_name=quora')",
    "print('Topic service status:', r.json().get('status'))",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 10 — View Topics
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-10", [
    "# CELL 10 — View All Topics ⭐",
    "import httpx",
    "",
    "r = httpx.get('http://localhost:8006/topics/all?dataset_name=quora')",
    "data = r.json()",
    "",
    "print(f\"{data['num_topics']} Topics Discovered in Quora Dataset:\\n\")",
    "print(f\"  {'ID':<5} {'Label':<35} {'Top 5 Words':<50} Docs\")",
    "print('  ' + '-' * 95)",
    "for t in data['topics']:",
    "    words = ', '.join(t['top_words'][:5])",
    "    print(f\"  {t['topic_id']:<5} {t['label']:<35} {words:<50} {t['doc_count']}\")",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 11 — Search with Topics
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-11", [
    "# CELL 11 — Search WITH Topic Labels ⭐",
    "import httpx",
    "",
    "QUERY = 'what is machine learning'",
    "",
    "# Step 1: Search via API Gateway",
    "r = httpx.post('http://localhost:8000/search', json={",
    "    'query': QUERY, 'dataset': 'quora', 'model': 'bm25',",
    "    'top_k': 5, 'use_refinement': False,",
    "}, timeout=30)",
    "result = r.json()",
    "",
    "# Step 2: Get topics for result docs (direct to topic service port 8006)",
    "doc_ids = [item['doc_id'] for item in result['results']]",
    "r2 = httpx.post('http://localhost:8006/topics/detect',",
    "    json={'dataset_name': 'quora', 'doc_ids': doc_ids}, timeout=30)",
    "topic_map = {item['doc_id']: item for item in r2.json().get('results', [])}",
    "",
    "print(f\"Query  : {result['query_original']}\")",
    "print(f\"Latency: {result['latency_ms']:.1f}ms\\n\")",
    "",
    "for item in result['results']:",
    "    topic = topic_map.get(item['doc_id'], {})",
    "    snippet = (item.get('snippet') or '')[:70].replace('\\n', ' ')",
    "    print(f\"  #{item['rank']}  doc={item['doc_id']}  score={item['score']:.4f}\")",
    "    print(f\"      Topic [{topic.get('topic_id','?')}]: {topic.get('topic_label','unknown')}\")",
    "    print(f\"      Words: {', '.join(topic.get('top_words', [])[:4])}\")",
    "    if snippet: print(f\"      Text : {snippet}...\")",
    "    print()",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 12 — Regular Search
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-12", [
    "# CELL 12 — Regular Search",
    "import httpx",
    "",
    "r = httpx.post('http://localhost:8000/search', json={",
    "    'query': 'what is machine learning',",
    "    'dataset': 'quora', 'model': 'bm25',",
    "    'top_k': 10, 'use_refinement': False,",
    "}, timeout=30)",
    "",
    "result = r.json()",
    "print(f\"Query  : {result['query_original']}\")",
    "print(f\"Model  : {result['model']}\")",
    "print(f\"Latency: {result['latency_ms']:.1f}ms\\n\")",
    "",
    "for item in result['results']:",
    "    snippet = (item.get('snippet') or '')[:80].replace('\\n', ' ')",
    "    print(f\"  #{item['rank']:<3} score={item['score']:.4f}  doc={item['doc_id']}\")",
    "    if snippet: print(f\"       {snippet}...\")",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 13 — Compare Models
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-13", [
    "# CELL 13 — Compare All 6 Search Models",
    "import httpx",
    "",
    "QUERY = 'what is machine learning'",
    "print('Query:', QUERY)",
    "print()",
    "print(f\"  {'Model':<22} {'Top Doc':<15} {'Score':<12} Latency\")",
    "print('  ' + '-' * 60)",
    "",
    "for model in ['tfidf', 'bm25', 'word2vec', 'bert', 'hybrid_serial', 'hybrid_parallel']:",
    "    try:",
    "        r = httpx.post('http://localhost:8000/search', json={",
    "            'query': QUERY, 'dataset': 'quora', 'model': model, 'top_k': 1",
    "        }, timeout=60)",
    "        d = r.json()",
    "        top = d['results'][0] if d['results'] else None",
    "        if top:",
    "            print(f\"  {model:<22} {top['doc_id']:<15} {top['score']:<12.4f} {d['latency_ms']:.0f}ms\")",
    "    except Exception as e:",
    "        print(f'  {model:<22} ERROR: {e}')",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 14 — Evaluate
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-14", [
    "# CELL 14 — Evaluate Models (MAP, nDCG@10)",
    "import httpx, math",
    "",
    "DATASET = 'quora'",
    "K = 10",
    "NUM_QUERIES = 50",
    "",
    "qrels = httpx.get(f'http://localhost:8001/dataset/qrels?dataset_name={DATASET}', timeout=120).json()['qrels']",
    "qtexts = {q['query_id']: q['original']",
    "          for q in httpx.get(f'http://localhost:8001/dataset/queries?dataset_name={DATASET}&offset=0&limit=500',",
    "                             timeout=60).json()['queries']}",
    "valid = [qid for qid in list(qrels.keys())[:NUM_QUERIES] if qid in qtexts]",
    "print(f'Evaluating {len(valid)} queries...')",
    "",
    "def ap(ranked, rel):",
    "    R = sum(1 for v in rel.values() if v >= 1)",
    "    if not R: return 0.0",
    "    h = s = 0",
    "    for i, d in enumerate(ranked, 1):",
    "        if rel.get(d, 0) >= 1: h += 1; s += h / i",
    "    return s / R",
    "",
    "def ndcg(ranked, rel, k):",
    "    dcg  = sum((2**rel.get(d,0)-1)/math.log2(i+1) for i,d in enumerate(ranked[:k],1) if rel.get(d,0)>0)",
    "    idcg = sum((2**r-1)/math.log2(i+1) for i,r in enumerate(sorted(rel.values(),reverse=True)[:k],1) if r>0)",
    "    return dcg/idcg if idcg else 0.0",
    "",
    "results = {}",
    "for model in ['tfidf', 'bm25', 'word2vec', 'bert']:",
    "    print(f'  Testing {model}...', end=' ', flush=True)",
    "    aps, ns = [], []",
    "    for qid in valid:",
    "        r = httpx.post('http://localhost:8000/search', json={",
    "            'query': qtexts[qid], 'dataset': DATASET, 'model': model,",
    "            'top_k': K, 'use_refinement': False}, timeout=30)",
    "        ranked = [x['doc_id'] for x in r.json().get('results', [])]",
    "        aps.append(ap(ranked, qrels[qid]))",
    "        ns.append(ndcg(ranked, qrels[qid], K))",
    "    n = max(len(aps), 1)",
    "    results[model] = {'map': round(sum(aps)/n, 4), 'ndcg': round(sum(ns)/n, 4)}",
    "    print(f\"MAP={results[model]['map']}  nDCG={results[model]['ndcg']}\")",
    "",
    "print()",
    "print(f\"  {'Model':<18} {'MAP':<12} nDCG@10\")",
    "print('  ' + '-' * 40)",
    "for m in sorted(results, key=lambda x: results[x]['map'], reverse=True):",
    "    print(f\"  {m:<18} {results[m]['map']:<12} {results[m]['ndcg']}\")",
    "best = max(results, key=lambda x: results[x]['map'])",
    "print(f'Best model: {best} (MAP={results[best][\"map\"]})')",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 15 — Logs
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-15", [
    "# CELL 15 — View Logs (Debug)",
    "import os",
    "",
    "logs = sorted([f for f in os.listdir('/content') if f.startswith('log_') and f.endswith('.txt')])",
    "for log in logs:",
    "    with open(f'/content/{log}') as f:",
    "        lines = f.read().strip().split('\\n')",
    "    recent = lines[-5:]",
    "    has_err = any('error' in l.lower() or 'traceback' in l.lower() for l in recent)",
    "    label = 'ERROR' if has_err else 'OK'",
    "    print(f'\\n[{label}]  {log}')",
    "    print('-' * 50)",
    "    print('\\n'.join(recent[-3:]))",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 16 — Save to Drive (includes LDA model)
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-16", [
    "# CELL 16 — Save ALL Indexes + LDA Model to Google Drive",
    "from google.colab import drive",
    "import shutil, os",
    "",
    "drive.mount('/content/drive')",
    "",
    "SRC = os.path.normpath(os.path.join(BACKEND_PATH, '..', 'data'))",
    "DST = '/content/drive/MyDrive/IR_Project_indexes'",
    "",
    "if os.path.exists(DST):",
    "    print('Removing old backup...')",
    "    shutil.rmtree(DST)",
    "",
    "print('Saving to Google Drive...')",
    "shutil.copytree(SRC, DST)",
    "",
    "# Verify LDA model is included",
    "lda_path = os.path.join(DST, 'quora', 'lda_model.pkl')",
    "if os.path.exists(lda_path):",
    "    print(f'  LDA model: {os.path.getsize(lda_path) // 1024 // 1024} MB')",
    "else:",
    "    print('  WARNING: LDA model not saved! Run Cell 9 first.')",
    "",
    "total_mb = sum(",
    "    os.path.getsize(os.path.join(root, f)) // 1024 // 1024",
    "    for root, dirs, files in os.walk(DST) for f in files)",
    "print(f'Total saved: ~{total_mb} MB')",
    "print('Next session: run Cell 16b to restore in ~1 min!')",
]))

# ─────────────────────────────────────────────────────────────────────────────
# CELL 16b — Restore from Drive
# ─────────────────────────────────────────────────────────────────────────────
nb["cells"].append(code_cell("cell-16b", [
    "# CELL 16b — Restore Indexes + LDA from Drive (fast start for next session)",
    "# Run this INSTEAD OF Cells 7+8+9 when you already have a backup",
    "from google.colab import drive",
    "import shutil, os, subprocess, time, httpx",
    "",
    "drive.mount('/content/drive')",
    "",
    "SRC = '/content/drive/MyDrive/IR_Project_indexes'",
    "DST = os.path.normpath(os.path.join(BACKEND_PATH, '..', 'data'))",
    "",
    "if not os.path.exists(SRC):",
    "    print('No Drive backup! Run Cells 7+8+9 first, then Cell 16.')",
    "else:",
    "    if os.path.exists(DST): shutil.rmtree(DST)",
    "    shutil.copytree(SRC, DST)",
    "    for dataset in os.listdir(DST):",
    "        files = os.listdir(os.path.join(DST, dataset))",
    "        has_lda = 'lda_model.pkl' in files",
    "        print(f'  Restored {dataset}: {len(files)} files  (LDA: {\"yes\" if has_lda else \"no\"})')",
    "    print('Restored!')",
    "",
    "    # Reload topic service so it picks up the LDA model",
    "    os.system('fuser -k 8006/tcp 2>/dev/null')",
    "    time.sleep(2)",
    "    env = os.environ.copy()",
    "    env['PYTHONPATH'] = BACKEND_PATH",
    "    log = open('/content/log_Topic.txt', 'w')",
    "    subprocess.Popen(['uvicorn', 'topic_service.main:app', '--host', '0.0.0.0',",
    "                      '--port', '8006', '--log-level', 'warning'],",
    "                     cwd=BACKEND_PATH, stdout=log, stderr=log, env=env)",
    "    time.sleep(8)",
    "    r = httpx.get('http://localhost:8006/topics/status?dataset_name=quora')",
    "    print('Topic status:', r.json().get('status'))",
    "    print('\\nDone! Go to Cell 11 to search with topics.')",
]))

# ─────────────────────────────────────────────────────────────────────────────
# WRITE
# ─────────────────────────────────────────────────────────────────────────────
out = r"C:\Users\n_alb\Downloads\IR_Project_Colab_v2.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Notebook saved: {out}")
print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])[:60].replace('\n',' ')
    print(f"  [{i}] {cell['cell_type']:8} {src}")
