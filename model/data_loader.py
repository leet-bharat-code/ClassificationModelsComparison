"""
Dataset loading and preprocessing for the classification task.
UCI Letter Recognition Dataset: 20,000 rows, 16 input features, 26 classes.
Source: https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data

Dataset handling:
- IF dataset file exists in data/: load from data/
- ELSE: download from UCI URL and save to data/, then load.
Reusable across training and Streamlit.
"""

import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

DATASET_FILENAME = "letter-recognition.data"
DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "letter-recognition/letter-recognition.data"
)

FEATURE_NAMES = [
    "xbox", "ybox", "width", "height", "onpix", "xbar", "ybar",
    "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "xege", "xegvy",
    "yege", "yegvx",
]
TARGET_NAME = "letter"

RANDOM_STATE = 42
TEST_SIZE = 0.25


def _get_data_dir():
    """Return absolute path to project data/ directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, "data")


def _get_dataset_path():
    """Return absolute path to the dataset file in data/."""
    return os.path.join(_get_data_dir(), DATASET_FILENAME)


def _download_dataset():
    """Download dataset from UCI URL and save to data/<dataset_file>."""
    data_dir = _get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    local_path = _get_dataset_path()
    urllib.request.urlretrieve(DATASET_URL, local_path)
    return local_path


def fetch_dataset() -> pd.DataFrame:
    """
    Load the UCI Letter Recognition dataset.
    IF file exists in data/: load from data/
    ELSE: download from UCI URL, save to data/, then load.
    """
    col_names = [TARGET_NAME] + FEATURE_NAMES
    local_path = _get_dataset_path()
    if os.path.isfile(local_path):
        return pd.read_csv(local_path, header=None, names=col_names)
    _download_dataset()
    return pd.read_csv(local_path, header=None, names=col_names)


def get_feature_target_split(df: pd.DataFrame):
    """Return (X, y, label_encoder)."""
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
    Returns dict with X_train, X_test, y_train, y_test, label_encoder, feature_names, etc.
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
    """Build a sklearn Pipeline for scaling."""
    return Pipeline([("scaler", StandardScaler())])


if __name__ == "__main__":
    splits = get_train_test_splits()
    print("Dataset loaded successfully.")
    print(f"Rows: {splits['n_rows']}, Features: {splits['n_features']}")
    print(f"Train size: {len(splits['y_train'])}, Test size: {len(splits['y_test'])}")
    print(f"Target variable: {splits['target_name']}")
