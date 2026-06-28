import json

filepath = r"C:\Users\n_alb\OneDrive\سطح المكتب\IR_Project\Final_Evaluation_and_Topics.ipynb"

with open(filepath, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        for i, line in enumerate(source):
            if 'num_queries' in line and '100000' in line:
                source[i] = '    "    \\"num_queries\\": 200, # عينة سريعة جداً\\n",\n'
                print("Found and replaced!")
            if '100000' in line:
                source[i] = line.replace('100000', '200')

with open(filepath, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("Done")
