# MS MARCO Colab Cells

## CELL 7b — Load MS MARCO Dataset (~5-10 min)
```python
import httpx, time

DATASET = 'msmarco'
r = httpx.post('http://localhost:8000/dataset/load',
               json={'dataset_name': DATASET}, timeout=30)
print('Response:', r.json())

while True:
    r = httpx.get(f'http://localhost:8000/dataset/status?dataset_name={DATASET}')
    d = r.json()
    status = d.get('status')
    docs = d.get('progress_docs', 0)
    total = d.get('total_docs', '?')
    print(f'  Status: {status:12}  Docs: {docs}/{total}')
    if status == 'ready':
        print(f'MS MARCO ready! ({total} docs)')
        break
    elif status == 'error':
        print('Error:', d.get('error'))
        break
    time.sleep(20)
```

## CELL 8b — Build MS MARCO Indexes (~30-60 min)
```python
import httpx, time

r = httpx.post('http://localhost:8000/index/build',
    json={'dataset_name': 'msmarco', 'models': ['tfidf', 'bm25', 'word2vec', 'bert']}, timeout=30)
print('Response:', r.json())

while True:
    r = httpx.get('http://localhost:8000/index/status?dataset_name=msmarco')
    d = r.json()
    status = d.get('status')
    built = d.get('built_models', [])
    print(f'  Status: {status:12}  Built: {built}')
    if status == 'ready':
        print('All MS MARCO indexes ready:', built)
        break
    elif status == 'error':
        print('Error:', d.get('error'))
        break
    time.sleep(20)
```

## CELL 9b — Build MS MARCO LDA (~5-10 min)
```python
import os, pickle, numpy as np
from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import ir_datasets

print('Loading MS MARCO corpus from ir_datasets...')
dataset = ir_datasets.load('beir/msmarco/test')

docs, doc_ids = [], []
for doc in dataset.docs_iter():
    text = (doc.text or '').strip()
    if text:
        docs.append(text)
        doc_ids.append(doc.doc_id)
    if len(docs) >= 200_000:   # use same 200K limit as indexing
        break
    if len(docs) % 50000 == 0 and len(docs) > 0:
        print(f'  Loaded {len(docs)} docs...')

print(f'Total: {len(docs)} documents')

os.system('rm -rf /tmp/joblib* 2>/dev/null')
print('Training LDA (10 topics)...')
vec = CountVectorizer(max_df=0.95, min_df=5, max_features=15_000, stop_words='english')
dtm = vec.fit_transform(docs)
lda = LatentDirichletAllocation(
    n_components=10, max_iter=15,
    learning_method='online', batch_size=512,
    random_state=42, n_jobs=1)
mat = lda.fit_transform(dtm)
print('Done!')

fn = vec.get_feature_names_out()
tids = np.argmax(mat, axis=1).tolist()
probs = mat.max(axis=1).tolist()
counts = [0]*10
for t in tids: counts[t] += 1

topics = []
for i, dist in enumerate(lda.components_):
    idx = dist.argsort()[-8:][::-1]; words = [fn[j] for j in idx]
    topics.append({'topic_id':i,'label':'/'.join(words[:3]),'top_words':words,'doc_count':counts[i]})

doc_map = {}
for i, did in enumerate(doc_ids):
    t = topics[tids[i]]
    doc_map[did] = {'topic_id':t['topic_id'],'topic_label':t['label'],'top_words':t['top_words'],'probability':round(float(probs[i]),4)}

for t in topics:
    print(f"  [{t['topic_id']}] {t['label']:30} ({t['doc_count']} docs)")

BACKEND_PATH = '/content/IR_Project/IR_Project/backend_services'
save_dir = Path(BACKEND_PATH).parent / 'data' / 'msmarco'
save_dir.mkdir(parents=True, exist_ok=True)
save_path = save_dir / 'lda_model.pkl'
with open(save_path, 'wb') as f:
    pickle.dump({'lda':lda,'vec':vec,'topics':topics,'map':doc_map,'n':10}, f)
print(f'Saved! ({save_path.stat().st_size//1024//1024} MB)')

# Reload topic service to pick up msmarco LDA
import subprocess, time, httpx
env = os.environ.copy(); env['PYTHONPATH'] = BACKEND_PATH
os.system('fuser -k 8006/tcp 2>/dev/null'); time.sleep(2)
log = open('/content/log_Topic.txt','w')
subprocess.Popen(['uvicorn','topic_service.main:app','--host','0.0.0.0','--port','8006','--log-level','warning'],
                 cwd=BACKEND_PATH, stdout=log, stderr=log, env=env)
time.sleep(8)
for ds in ['quora', 'msmarco']:
    r = httpx.get(f'http://localhost:8006/topics/status?dataset_name={ds}')
    print(f'  {ds}: {r.json().get("status")}')
```

## CELL 11b — Test MS MARCO Search
```python
import httpx

r = httpx.post('http://localhost:8000/search', json={
    'query': 'what is machine learning',
    'dataset': 'msmarco', 'model': 'bm25',
    'top_k': 5, 'use_refinement': False,
}, timeout=30)
result = r.json()

doc_ids = [item['doc_id'] for item in result['results']]
r2 = httpx.post('http://localhost:8006/topics/detect',
    json={'dataset_name': 'msmarco', 'doc_ids': doc_ids}, timeout=30)
topic_map = {item['doc_id']: item for item in r2.json().get('results', [])}

print(f"Query  : {result['query_original']}")
print(f"Latency: {result['latency_ms']:.1f}ms\n")
for item in result['results']:
    topic = topic_map.get(item['doc_id'], {})
    print(f"  #{item['rank']}  doc={item['doc_id']}  score={item['score']:.4f}")
    print(f"      Topic [{topic.get('topic_id','?')}]: {topic.get('topic_label','unknown')}")
    print()
```

## Updated CELL 16 — Save BOTH datasets to Drive
```python
from google.colab import drive
import shutil, os

drive.mount('/content/drive')

BACKEND_PATH = '/content/IR_Project/IR_Project/backend_services'
SRC = os.path.normpath(os.path.join(BACKEND_PATH, '..', 'data'))
DST = '/content/drive/MyDrive/IR_Project_indexes'

if os.path.exists(DST): shutil.rmtree(DST)
shutil.copytree(SRC, DST)

for ds in ['quora', 'msmarco']:
    ds_path = os.path.join(DST, ds)
    if os.path.exists(ds_path):
        files = os.listdir(ds_path)
        has_lda = 'lda_model.pkl' in files
        total_mb = sum(os.path.getsize(os.path.join(ds_path,f))//1024//1024 for f in files)
        print(f'  {ds}: {len(files)} files, ~{total_mb} MB  (LDA: {"yes" if has_lda else "no"})')

print('Both datasets saved to Drive!')
```
