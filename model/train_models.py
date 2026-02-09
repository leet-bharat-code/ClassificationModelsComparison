"""
Training logic for all six classification models.
Uses sklearn Pipelines for preprocessing + model. Saves fitted pipelines and
evaluation results. Training logic is NOT in app.py.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from model.data_loader import get_train_test_splits, FEATURE_NAMES
from model.evaluate import compute_all_metrics, metrics_to_dataframe

RANDOM_STATE = 42
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
RESULTS_FILENAME = "evaluation_results.csv"


def _ensure_artifacts_dir():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def _build_preprocessing():
    return Pipeline([("scaler", StandardScaler())])


def _get_models():
    """Return dict of (name -> sklearn/XGBoost estimator). All use fixed random_state where applicable."""
    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
        ),
    }


def train_all_models():
    """
    Load data, train all six models with preprocessing pipeline,
    evaluate on the same test set, save pipelines and results.
    Returns (results_df, label_encoder, n_classes).
    """
    _ensure_artifacts_dir()
    splits = get_train_test_splits(random_state=RANDOM_STATE)
    X_train = splits["X_train"]
    X_test = splits["X_test"]
    y_train = splits["y_train"]
    y_test = splits["y_test"]
    label_encoder = splits["label_encoder"]
    n_classes = len(np.unique(y_train))

    preproc = _build_preprocessing()
    X_train_scaled = preproc.fit_transform(X_train)
    X_test_scaled = preproc.transform(X_test)

    models = _get_models()
    metrics_list = []

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocessor", _build_preprocessing().fit(X_train, y_train)),
            ("classifier", estimator),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)

        metrics = compute_all_metrics(
            y_test, y_pred, y_proba, n_classes, model_name=name
        )
        metrics_list.append(metrics)

        safe_name = name.replace(" ", "_").lower()
        path = os.path.join(ARTIFACTS_DIR, f"pipeline_{safe_name}.joblib")
        joblib.dump(pipe, path)

    joblib.dump(label_encoder, os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
    joblib.dump(preproc, os.path.join(ARTIFACTS_DIR, "preprocessor.joblib"))
    joblib.dump({"n_classes": n_classes, "feature_names": FEATURE_NAMES}, 
                os.path.join(ARTIFACTS_DIR, "meta.joblib"))

    results_df = metrics_to_dataframe(metrics_list)
    results_path = os.path.join(ARTIFACTS_DIR, RESULTS_FILENAME)
    results_df.to_csv(results_path, index=False)

    return results_df, label_encoder, n_classes


if __name__ == "__main__":
    results_df, le, n_classes = train_all_models()
    print("Training complete. Evaluation results:")
    print(results_df.to_string())
