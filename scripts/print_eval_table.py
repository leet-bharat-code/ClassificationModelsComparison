"""
Print the evaluation results as a Markdown table for pasting into README.
Run from project root after training: python scripts/print_eval_table.py
"""
import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "model", "artifacts", "evaluation_results.csv")

if not os.path.isfile(CSV_PATH):
    print("Run training first: python -m model.train_models", file=sys.stderr)
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
cols = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
df = df[cols]
for col in cols[1:]:
    df[col] = df[col].apply(lambda x: f"{x:.4f}")
header = "| " + " | ".join(cols) + " |"
sep = "|" + "|".join(["---"] * len(cols)) + "|"
lines = [header, sep]
for _, row in df.iterrows():
    lines.append("| " + " | ".join(str(v) for v in row) + " |")
print("\n".join(lines))
