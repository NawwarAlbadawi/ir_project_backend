import json

filepath = r"C:\Users\n_alb\OneDrive\سطح المكتب\IR_Project\Final_Evaluation_and_Topics.ipynb"

with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell 5 (the topics cell)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        for i, line in enumerate(source):
            if "weights = list(range(10, 0, -1))" in line:
                source[i] = line.replace("10", "len(top_words)")
                print("Replaced weights successfully!")

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook patched for dynamic weights.")
