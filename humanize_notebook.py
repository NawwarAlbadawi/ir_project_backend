import json

filepath = r"C:\Users\n_alb\OneDrive\سطح المكتب\IR_Project\Final_Evaluation_and_Topics.ipynb"

with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update cell 0 (Markdown intro)
nb["cells"][0]["source"] = [
    "# Information Retrieval System - Final Evaluation\n",
    "هذا الملف يوضح عملية التقييم لمحرك البحث الذي قمنا ببنائه. سنقوم باختبار النظام باستخدام المقاييس المعتمدة مثل MAP و nDCG.\n",
    "بالإضافة إلى ذلك، نعرض هنا نتائج خوارزمية LDA لاستخراج المواضيع (Topic Modeling)."
]

# Update cell 2 (Markdown eval)
nb["cells"][2]["source"] = [
    "## 1. System Evaluation\n",
    "سنقوم الآن باختبار نموذج `bm25` مع مقارنة الأداء قبل وبعد تفعيل ميزة تحسين الاستعلام (Query Refinement).\n",
    "أخذنا عينة من الاستعلامات لاختبار النظام ورسم النتائج."
]

# Update cell 3 (Code eval)
source = nb["cells"][3]["source"]
for i, line in enumerate(source):
    if 'print("⏳' in line:
        source[i] = 'print("Starting evaluation process...")\n'
    if 'print(f"✅' in line:
        source[i] = '    print("Evaluation completed successfully.")\n'
    if 'print(f"📌' in line:
        source[i] = '    print(f"Evaluated on {num_q} queries.")\n'

# Update cell 4 (Markdown topics)
nb["cells"][4]["source"] = [
    "## 2. Topic Modeling Visualization\n",
    "قراءة النموذج المدرب وعرض المواضيع المكتشفة مع الكلمات المفتاحية الخاصة بها."
]

# Update cell 5 (Code topics)
source2 = nb["cells"][5]["source"]
for i, line in enumerate(source2):
    if 'print(f"❌' in line:
        source2[i] = '    print(f"Model not found at {lda_path}")\n'
    if 'print(f"📌' in line:
        source2[i] = '    print(f"Number of topics: {len(topics)}")\n'

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Notebook humanized successfully")
