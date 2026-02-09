"""
Print evaluation results as a markdown-ready table to stdout.
Copy-paste ready for README.md Section 3.
If data/evaluation_results.csv does not exist, run training first (writes CSV), then print.
PURE CLI script — no Streamlit.
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "evaluation_results.csv")
COLS = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def main():
    if not os.path.isfile(CSV_PATH):
        sys.stderr.write("Metrics CSV not found. Running training to create it...\n")
        sys.path.insert(0, PROJECT_ROOT)
        from model.train_models import train_all_models
        train_all_models(save_artifacts=False)
    if not os.path.isfile(CSV_PATH):
        sys.stderr.write("Training did not produce metrics file. Aborting.\n")
        sys.exit(1)
    df = pd.read_csv(CSV_PATH)
    df = df[COLS]
    for col in COLS[1:]:
        df[col] = df[col].apply(lambda x: f"{float(x):.4f}")
    header = "| " + " | ".join(COLS) + " |"
    sep = "|" + "|".join(["---"] * len(COLS)) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
