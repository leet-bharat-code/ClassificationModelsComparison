"""
Evaluation metrics computation for classification models.
Computes Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient
using sklearn.metrics only. Results are stored in a structured pandas DataFrame.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize


def compute_auc_multiclass(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    """
    Compute AUC for multiclass using One-vs-Rest with macro averaging.
    y_proba: shape (n_samples, n_classes).
    """
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    try:
        auc = roc_auc_score(y_true_bin, y_proba, average="macro")
    except ValueError:
        auc = 0.0
    return float(auc)


def compute_auc_binary(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute AUC for binary classification using roc_auc_score."""
    if y_proba.ndim > 1:
        y_proba = y_proba[:, 1]
    try:
        return float(roc_auc_score(y_true, y_proba))
    except ValueError:
        return 0.0


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int,
    model_name: str,
    average: str = "macro",
) -> dict:
    """
    Compute all required metrics for a single model.
    For multiclass: precision, recall, f1 use average='macro' by default.
    """
    is_binary = n_classes == 2
    if is_binary:
        auc = compute_auc_binary(y_true, y_proba)
    else:
        auc = compute_auc_multiclass(y_true, y_proba, n_classes)

    return {
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc,
        "Precision": float(
            precision_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true, y_pred, average=average, zero_division=0)
        ),
        "F1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def metrics_to_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """Build a structured pandas DataFrame from a list of metric dicts."""
    return pd.DataFrame(metrics_list)
