"""
Training logic for all six classification models.
Uses model/registry for estimators. Trains in memory by default.
Metrics persisted to data/evaluation_results.csv (lightweight).
Optional save_artifacts flag to write .joblib to model/artifacts/.
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model.data_loader import get_train_test_splits, FEATURE_NAMES, _get_data_dir
from model.evaluate import compute_all_metrics, metrics_to_dataframe
from model.registry import get_estimators

RANDOM_STATE = 42
METRICS_FILENAME = "evaluation_results.csv"


def _get_metrics_path():
    """Path to metrics CSV in data/ (lightweight, no .joblib)."""
    return os.path.join(_get_data_dir(), METRICS_FILENAME)


def _build_preprocessing():
    return Pipeline([("scaler", StandardScaler())])


def train_all_models(save_artifacts: bool = False):
    """
    Train all six models on the same dataset, compute metrics, persist metrics CSV.
    IF save_artifacts=False (default): train in memory, write metrics to data/evaluation_results.csv, return objects.
    IF save_artifacts=True: also save pipelines/label_encoder/meta to model/artifacts/ (optional).

    Returns:
        metrics_df: pandas DataFrame with columns Model | Accuracy | AUC | Precision | Recall | F1 | MCC
        pipelines: dict of (model_name -> fitted Pipeline)
        label_encoder: fitted LabelEncoder
        meta: dict with n_classes, feature_names
    """
    splits = get_train_test_splits(random_state=RANDOM_STATE)
    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    label_encoder = splits["label_encoder"]
    n_classes = len(np.unique(y_train))

    estimators = get_estimators()
    metrics_list = []
    pipelines = {}

    for name, estimator in estimators.items():
        pipe = Pipeline([
            ("preprocessor", _build_preprocessing()),
            ("classifier", estimator),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        metrics = compute_all_metrics(
            y_test, y_pred, y_proba, n_classes, model_name=name
        )
        metrics_list.append(metrics)
        pipelines[name] = pipe

    metrics_df = metrics_to_dataframe(metrics_list)
    meta = {"n_classes": n_classes, "feature_names": FEATURE_NAMES}

    # Persist metrics in lightweight format (CSV in data/)
    os.makedirs(_get_data_dir(), exist_ok=True)
    metrics_df.to_csv(_get_metrics_path(), index=False)

    if save_artifacts:
        import joblib
        artifacts_dir = os.path.join(os.path.dirname(__file__), "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        for name, pipe in pipelines.items():
            safe = name.replace(" ", "_").lower()
            joblib.dump(pipe, os.path.join(artifacts_dir, f"pipeline_{safe}.joblib"))
        joblib.dump(label_encoder, os.path.join(artifacts_dir, "label_encoder.joblib"))
        joblib.dump(meta, os.path.join(artifacts_dir, "meta.joblib"))

    return metrics_df, pipelines, label_encoder, meta


if __name__ == "__main__":
    metrics_df, pipelines, le, meta = train_all_models(save_artifacts=False)
    print("Training complete. Evaluation results:")
    print(metrics_df.to_string())
    print(f"\nMetrics saved to {_get_metrics_path()}")
