"""
Model factory / registry: provides all six required classifiers for training.
Used by train_models.py. No training logic here.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

RANDOM_STATE = 42

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Naive Bayes",
    "Random Forest",
    "XGBoost",
]


def get_estimators():
    """
    Return a dict of (model_name -> estimator instance) for all six models.
    Each call returns fresh instances. All use fixed random_state where applicable.
    """
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
