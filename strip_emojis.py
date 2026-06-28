import json

filepath = r"C:\Users\n_alb\OneDrive\سطح المكتب\IR_Project\Final_Evaluation_and_Topics.ipynb"

with open(filepath, "r", encoding="utf-8") as f:
    text = f.read()

# Remove the specific emojis causing the crash
text = text.replace(r"\u274c", "")
text = text.replace(r"\ud83d\udccc", "")
text = text.replace(r"\u23f3", "")
text = text.replace(r"\u2705", "")

with open(filepath, "w", encoding="utf-8") as f:
    f.write(text)

print("Notebook emojis stripped successfully")
