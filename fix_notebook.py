import json

with open(r'C:\Users\n_alb\Downloads\IR_Project_Colab_v2.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

nltk_block = (
    "\n# Download NLTK data (needed by Preprocessing & Refinement services)\n"
    "import nltk\n"
    "for pkg in ['punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger_eng', 'omw-1.4']:\n"
    "    nltk.download(pkg, quiet=True)\n"
    "print('NLTK data ready')\n"
)

for cell in nb['cells']:
    if cell.get('id') == 'cell-03':
        marker = "# IMPORTANT: Set PYTHONPATH so subprocesses can find modules"
        cell['source'] = cell['source'].replace(marker, nltk_block + marker)
        print("Cell 3 updated")
        break

with open(r'C:\Users\n_alb\Downloads\IR_Project_Colab_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Saved!")
