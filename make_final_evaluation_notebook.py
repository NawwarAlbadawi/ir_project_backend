import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# 📊 IR System Evaluation & Topic Modeling Analysis
هذا الدفتر (Notebook) مخصص لتقييم أداء محرك البحث باستخدام المقاييس القياسية (MAP, nDCG) على جميع الاستعلامات الموجودة في ملف الـ qrels. 
كما يتضمن عرضاً بيانياً لنتائج نمذجة المواضيع (Topic Modeling) باستخدام خوارزمية LDA كما طلب الدكتور.
"""

code_setup = """\
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

API_URL = "http://localhost:8000"
DATASET = "quora"
"""

text_eval = """\
## 1. التقييم الشامل (Full Evaluation)
سيقوم هذا الكود بإرسال طلب إلى الـ API لتقييم نموذج `bm25` مع ميزة التحسين `Compare Refinement`.
التقييم سيشمل **جميع** الكويريات الموجودة في ملف الـ qrels (سنضع num_queries = 100000 لضمان تقييم كل شيء).
"""

code_eval = """\
print("⏳ جاري تقييم جميع الكويريات في الـ qrels. قد يستغرق هذا بضع دقائق...")

payload = {
    "dataset": DATASET,
    "model": "bm25",
    "num_queries": 100000, # رقم ضخم لضمان تقييم كل الاستعلامات في qrels
    "k": 10,
    "compare_refinement": True
}

response = requests.post(f"{API_URL}/search/evaluate", json=payload)
if response.status_code == 200:
    results = response.json()
    base = results["base"]
    refined = results["refined"]
    
    num_q = base["num_queries"]
    print(f"✅ تمت عملية التقييم بنجاح!")
    print(f"📌 عدد الاستعلامات (Queries) التي تم استخدامها في التقييم من ملف الـ qrels هو: {num_q} استعلام.")
    
    # تحضير البيانات للرسم
    metrics = ['MAP', 'Recall', 'Precision@10', 'nDCG@10']
    base_scores = [base['map'], base['mean_recall'], base['mean_precision_at_k'], base['mean_ndcg_at_k']]
    refined_scores = [refined['map'], refined['mean_recall'], refined['mean_precision_at_k'], refined['mean_ndcg_at_k']]
    
    # رسم بياني للمقارنة
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, base_scores, width, label='Base (BM25)', color='skyblue')
    ax.bar(x + width/2, refined_scores, width, label='Refined (BM25 + Synonyms)', color='lightgreen')

    ax.set_ylabel('Scores')
    ax.set_title('Evaluation Metrics: Base vs Refined (ALL QRELS)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()
else:
    print("❌ خطأ في التقييم:", response.text)
"""

text_topics = """\
## 2. الرسومات البيانية لنمذجة المواضيع (Topic Detection Charts)
الآن سنقوم بقراءة النموذج المُدرب `lda_model.pkl` وعرض المواضيع التي استنتجها الكود، مع رسم بياني يوضح الكلمات المفتاحية الأكثر تأثيراً في كل موضوع (Topic Chart).
"""

code_topics = """\
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# مسار نموذج الـ LDA لمجموعة بيانات Quora
lda_path = Path("data") / "quora" / "lda_model.pkl"

if not lda_path.exists():
    print(f"❌ لم يتم العثور على النموذج في {lda_path}")
else:
    with open(lda_path, 'rb') as f:
        lda_data = pickle.load(f)
    
    # الكود الخاص بنا يحفظ المواضيع داخل self._topics، وعندما نقوم بعمل pickle يحفظ الـ object
    # سنصل إلى المواضيع
    topics = lda_data._topics
    
    print(f"📌 عدد المواضيع (Topics) المستخرجة هو: {len(topics)} مواضيع.")
    
    # سنقوم برسم أول 6 مواضيع كعينة توضيحية
    num_to_plot = min(6, len(topics))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True)
    axes = axes.flatten()

    for i in range(num_to_plot):
        top_words = topics[i]['top_words'][:10] # أهم 10 كلمات
        # أوزان وهمية للرسم التوضيحي بما أننا خزننا الكلمات فقط (من الأهم للأقل أهمية)
        weights = list(range(10, 0, -1)) 
        
        ax = axes[i]
        ax.barh(top_words, weights, color='salmon')
        ax.set_title(topics[i]['label'], fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlabel("Importance Weight")

    plt.suptitle("Top Words per Topic (LDA Topic Modeling)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_markdown_cell(text_eval),
    nbf.v4.new_code_cell(code_eval),
    nbf.v4.new_markdown_cell(text_topics),
    nbf.v4.new_code_cell(code_topics)
]

with open('Final_Evaluation_and_Topics.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("✅ Created notebook: Final_Evaluation_and_Topics.ipynb")
