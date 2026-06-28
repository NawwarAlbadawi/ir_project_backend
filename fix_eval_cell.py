import json

filepath = r"C:\Users\n_alb\OneDrive\سطح المكتب\IR_Project\Final_Evaluation_and_Topics.ipynb"

with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Loop over all cells and remove the exact emoji strings
for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        source = cell["source"]
        for i, line in enumerate(source):
            if "⏳" in line or "جاري تقييم جميع الكويريات" in line:
                source[i] = 'print("Starting evaluation process...")\n'
                print("Replaced ⏳ line")
            elif "✅" in line or "تمت عملية التقييم بنجاح" in line:
                source[i] = '    print("Evaluation completed successfully.")\n'
                print("Replaced ✅ line")
            elif "📌" in line or "عدد الاستعلامات" in line:
                # Be careful not to replace the topic count line if we don't want to, but we can replace both safely.
                if "qrels" in line:
                    source[i] = '    print(f"Evaluated on {num_q} queries.")\n'
                    print("Replaced 📌 qrels line")
                elif "مواضيع" in line:
                    source[i] = '    print(f"Number of topics: {len(topics)}")\n'
                    print("Replaced 📌 topics line")
            elif "❌" in line or "لم يتم العثور" in line:
                source[i] = '    print(f"Model not found at {lda_path}")\n'
                print("Replaced ❌ line")

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook evaluation cell patched successfully.")
