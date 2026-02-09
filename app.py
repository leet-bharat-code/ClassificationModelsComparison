"""
Streamlit entry point ONLY. No training logic here.
Provides: CSV upload (test data), model selection, metrics display,
confusion matrix / classification report visualization.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Resolve paths relative to project root (parent of app.py)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "model", "artifacts")
RESULTS_CSV = os.path.join(ARTIFACTS_DIR, "evaluation_results.csv")


@st.cache_resource
def load_artifacts():
    """Load saved pipelines, label encoder, and metadata once."""
    meta = joblib.load(os.path.join(ARTIFACTS_DIR, "meta.joblib"))
    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.joblib"))
    feature_names = meta["feature_names"]
    n_classes = meta["n_classes"]
    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost",
    ]
    pipelines = {}
    for name in model_names:
        safe = name.replace(" ", "_").lower()
        path = os.path.join(ARTIFACTS_DIR, f"pipeline_{safe}.joblib")
        if os.path.isfile(path):
            pipelines[name] = joblib.load(path)
    return {
        "pipelines": pipelines,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
        "n_classes": n_classes,
    }


@st.cache_data
def load_evaluation_results():
    """Load the evaluation results DataFrame (feeds README and UI)."""
    if not os.path.isfile(RESULTS_CSV):
        return None
    return pd.read_csv(RESULTS_CSV)


def ensure_trained():
    """If artifacts missing, run training once."""
    if not os.path.isdir(ARTIFACTS_DIR) or not os.path.isfile(RESULTS_CSV):
        with st.spinner("Training models (first run)..."):
            try:
                from model.train_models import train_all_models
                train_all_models()
            except Exception as e:
                st.error(
                    "Training failed (often due to no network). "
                    "Download the UCI Letter Recognition dataset and place "
                    "letter-recognition.data in the data/ folder, or run with network. "
                    f"Error: {e}"
                )
                st.stop()
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()


def validate_upload_schema(uploaded_df: pd.DataFrame, expected_features: list) -> tuple[bool, str]:
    """Validate that uploaded CSV has the expected feature columns. Returns (ok, message)."""
    missing = set(expected_features) - set(uploaded_df.columns)
    if missing:
        return False, f"Missing columns: {sorted(missing)}. Expected: {expected_features}"
    extra = set(uploaded_df.columns) - set(expected_features)
    if extra:
        # Allow extra columns (e.g. target) but require at least expected features
        pass
    return True, "Schema OK"


def main():
    st.set_page_config(page_title="Classification Models", layout="wide")
    st.title("Classification Models — Evaluation & Inference")

    ensure_trained()
    artifacts = load_artifacts()
    pipelines = artifacts["pipelines"]
    label_encoder = artifacts["label_encoder"]
    feature_names = list(artifacts["feature_names"])
    n_classes = artifacts["n_classes"]

    results_df = load_evaluation_results()
    if results_df is None:
        st.warning("Evaluation results not found. Run training first.")
        return

    # ---- Model selection ----
    selected_model = st.selectbox(
        "Select model",
        options=list(pipelines.keys()),
        index=0,
    )

    # ---- Evaluation metrics display (from stored results) ----
    st.subheader("Evaluation metrics")
    row = results_df[results_df["Model"] == selected_model].iloc[0]
    cols = st.columns(6)
    for i, metric in enumerate(["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        cols[i].metric(metric, f"{row[metric]:.4f}")

    # ---- Confusion Matrix OR Classification Report ----
    st.subheader("Confusion matrix and classification report")
    test_preds = get_test_predictions_cached(selected_model)
    if test_preds is not None:
        y_true, y_pred = test_preds["y_true"], test_preds["y_pred"]
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {selected_model}")
        st.pyplot(fig)
        plt.close(fig)
        report = classification_report(
            y_true, y_pred, target_names=list(label_encoder.classes_), output_dict=True
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    else:
        st.info("Confusion matrix and classification report require test set predictions (run once).")

    # ---- CSV upload: Upload TEST data only ----
    st.subheader("Upload TEST data only")
    st.caption("Upload TEST data only. Use a CSV with the same feature columns as the training data. Labels are optional for inference.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
        else:
            ok, msg = validate_upload_schema(uploaded_df, feature_names)
            if not ok:
                st.error(msg)
            else:
                X_upload = uploaded_df[feature_names].copy()
                pipe = pipelines[selected_model]
                preds = pipe.predict(X_upload)
                pred_labels = label_encoder.inverse_transform(preds)
                result_upload = pd.DataFrame({"predicted_label": pred_labels, "predicted_class": preds})
                st.write("Predictions:")
                st.dataframe(result_upload, use_container_width=True)


@st.cache_data
def get_test_predictions_cached(model_name: str):
    """Get y_true, y_pred for the fixed test set for the selected model (cached)."""
    from model.data_loader import get_train_test_splits
    splits = get_train_test_splits()
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    safe = model_name.replace(" ", "_").lower()
    path = os.path.join(ARTIFACTS_DIR, f"pipeline_{safe}.joblib")
    if not os.path.isfile(path):
        return None
    pipe = joblib.load(path)
    y_pred = pipe.predict(X_test)
    return {"y_true": y_test, "y_pred": y_pred}


if __name__ == "__main__":
    main()
