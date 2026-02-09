Dataset and metrics (no model artifacts committed).

- letter-recognition.data: UCI Letter Recognition dataset.
  If missing: first run downloads from UCI and saves here.
  For offline use: download and place this file here.

- evaluation_results.csv: Written by python -m model.train_models.
  Used by scripts/print_eval_table.py and by the app (via in-memory training).

No .joblib/.pkl files are stored here; models are created at runtime.
