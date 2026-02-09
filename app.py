"""
Streamlit entry point ONLY. No training code inside UI logic.
Uses shared functions from model/. On first run: if trained models not in memory,
trains via model.train_models and caches with st.cache_resource.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


@st.cache_resource
def get_trained_models():
    """
    Train all models in memory (no .joblib load). Cache result.
    Returns dict with pipelines, label_encoder, feature_names, n_classes, metrics_df.
    """
    from model.train_models import train_all_models
    metrics_df, pipelines, label_encoder, meta = train_all_models(save_artifacts=False)
    return {
        "metrics_df": metrics_df,
        "pipelines": pipelines,
        "label_encoder": label_encoder,
        "feature_names": meta["feature_names"],
        "n_classes": meta["n_classes"],
    }


def validate_upload_schema(uploaded_df: pd.DataFrame, expected_features: list) -> tuple[bool, str]:
    """Validate that uploaded CSV has the expected feature columns."""
    missing = set(expected_features) - set(uploaded_df.columns)
    if missing:
        return False, f"Missing columns: {sorted(missing)}. Expected: {expected_features}"
    return True, "Schema OK"


def main():
    st.set_page_config(page_title="Classification Models", layout="wide")
    st.title("Classification Models — Evaluation & Inference")

    with st.spinner("Training models (first run)..."):
        try:
            result = get_trained_models()
        except Exception as e:
            st.error(
                "Training failed. Ensure the dataset is available: "
                "place letter-recognition.data in data/ for offline use, "
                "or run with network so it can download. "
                f"Error: {e}"
            )
            st.stop()

    pipelines = result["pipelines"]
    label_encoder = result["label_encoder"]
    feature_names = list(result["feature_names"])
    metrics_df = result["metrics_df"]

    # ---- Model selection ----
    selected_model = st.selectbox(
        "Select model",
        options=list(pipelines.keys()),
        index=0,
    )

    # ---- Evaluation metrics (all 6) ----
    st.subheader("Evaluation metrics")
    row = metrics_df[metrics_df["Model"] == selected_model].iloc[0]
    cols = st.columns(6)
    for i, metric in enumerate(["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]):
        cols[i].metric(metric, f"{row[metric]:.4f}")

    # ---- Confusion matrix and classification report ----
    st.subheader("Confusion matrix and classification report")
    from model.data_loader import get_train_test_splits
    splits = get_train_test_splits(random_state=42)
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    pipe = pipelines[selected_model]
    y_pred = pipe.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {selected_model}")
    st.pyplot(fig)
    plt.close(fig)
    report = classification_report(
        y_test, y_pred, target_names=list(label_encoder.classes_), output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    # ---- CSV upload: TEST DATA ONLY ----
    st.subheader("Upload TEST data only")
    st.caption("Upload TEST data only. CSV must have the same feature columns as the training data. Labels optional.")
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


if __name__ == "__main__":
    main()
