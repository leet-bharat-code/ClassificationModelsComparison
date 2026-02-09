"""
Dataset loading and preprocessing for the classification task.
UCI Letter Recognition Dataset: 20,000 rows, 16 input features, 26 classes.
Source: https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

# Dataset source URL (UCI ML Repository)
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "letter-recognition/letter-recognition.data"
)

# Column names: first column is target (letter), rest are 16 features
FEATURE_NAMES = [
    "xbox", "ybox", "width", "height", "onpix", "xbar", "ybar",
    "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "xege", "xegvy",
    "yege", "yegvx",
]
TARGET_NAME = "letter"

RANDOM_STATE = 42
TEST_SIZE = 0.25


def fetch_dataset() -> pd.DataFrame:
    """Load the UCI Letter Recognition dataset from URL or local data/letter-recognition.data."""
    import os
    col_names = [TARGET_NAME] + FEATURE_NAMES
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(os.path.dirname(script_dir), "data", "letter-recognition.data")
    if os.path.isfile(local_path):
        df = pd.read_csv(local_path, header=None, names=col_names)
        return df
    df = pd.read_csv(DATASET_URL, header=None, names=col_names)
    return df


def get_feature_target_split(df: pd.DataFrame):
    """Return (X, y) with feature matrix and encoded target."""
    X = df[FEATURE_NAMES].copy()
    y_raw = df[TARGET_NAME].copy()
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le


def get_train_test_splits(
    random_state: int = RANDOM_STATE,
    test_size: float = TEST_SIZE,
):
    """
    Load dataset, split into train/test with fixed random_state.
    Returns X_train, X_test, y_train, y_test, label_encoder, feature_names.
    """
    df = fetch_dataset()
    X, y, label_encoder = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "label_encoder": label_encoder,
        "feature_names": FEATURE_NAMES,
        "target_name": TARGET_NAME,
        "n_rows": len(df),
        "n_features": len(FEATURE_NAMES),
    }


def build_preprocessing_pipeline():
    """Build a sklearn Pipeline for scaling (no encoding needed for features; they are numeric)."""
    return Pipeline([("scaler", StandardScaler())])


if __name__ == "__main__":
    # Quick sanity check and documentation output
    splits = get_train_test_splits()
    print("Dataset loaded successfully.")
    print(f"Rows: {splits['n_rows']}, Features: {splits['n_features']}")
    print(f"Train size: {len(splits['y_train'])}, Test size: {len(splits['y_test'])}")
    print(f"Target variable: {splits['target_name']}")
