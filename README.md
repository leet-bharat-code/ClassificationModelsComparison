# Classification Models — Academic Submission

**How to run:** Install dependencies (`pip install -r requirements.txt`), then run training once (requires network to download the UCI dataset, or place `letter-recognition.data` in `data/`): `python -m model.train_models`. Then start the app: `streamlit run app.py`. To get the evaluation table for Section 3, run `python scripts/print_eval_table.py` after training and paste the output into the table below.

---

## 1. Problem Statement

This project addresses a **multiclass classification** task: predicting the identity of a capital letter (A–Z) from 16 numerical features derived from raster scan images of the letters. The goal is to train and compare six classifiers on the same dataset and evaluate them using a fixed test set and a common set of metrics. The task is well-defined, with 26 classes and continuous input features, and is suitable for comparing linear, tree-based, and ensemble methods.

---

## 2. Dataset Description

- **Dataset source:** UCI Machine Learning Repository — Letter Recognition Data Set  
  URL: https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data

- **Number of rows:** 20,000

- **Number of features:** 16 input features (xbox, ybox, width, height, onpix, xbar, ybar, x2bar, y2bar, xybar, x2ybr, xy2br, xege, xegvy, yege, yegvx). All are integer-valued attributes derived from character images.

- **Target variable:** `letter` — capital letter A through Z (26 classes).

The data is split into train and test sets with a fixed `random_state=42` and 75% train / 25% test. Preprocessing includes scaling via `StandardScaler` inside sklearn Pipelines. No encoding is required for the target beyond label encoding (A–Z to 0–25), as all inputs are numeric.

---

## 3. Models Used and Evaluation

All models are trained on the same preprocessed data and evaluated on the same test set. Metrics are computed with `sklearn.metrics` and stored in a pandas DataFrame (and in `model/artifacts/evaluation_results.csv`). The comparison table below has the required columns: Model | Accuracy | AUC | Precision | Recall | F1 | MCC.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|------------|--------|-----|-----|
| Logistic Regression | (run training) | (run training) | (run training) | (run training) | (run training) | (run training) |
| Decision Tree | (run training) | (run training) | (run training) | (run training) | (run training) | (run training) |
| K-Nearest Neighbors | (run training) | (run training) | (run training) | (run training) | (run training) | (run training) |
| Naive Bayes | (run training) | (run training) | (run training) | (run training) | (run training) | (run training) |
| Random Forest | (run training) | (run training) | (run training) | (run training) | (run training) | (run training) |
| XGBoost | (run training) | (run training) | (run training) | (run training) | (run training) | (run training) |

**Note:** Run `python -m model.train_models` from the project root (with network access for first-time dataset download) to generate `model/artifacts/evaluation_results.csv`. That CSV contains the exact values for all six models and all six metrics; you can paste them into this table for your submission.

---

## 4. Observations

- **Logistic Regression:** A linear model that learns decision boundaries in the scaled feature space. It typically achieves moderate accuracy on Letter Recognition because the problem is not linearly separable; some letter pairs (e.g. O vs Q) are similar in the given features. It is stable and has low variance but can underfit, leading to lower AUC and recall on harder classes. Bias-variance: higher bias, lower variance.

- **Decision Tree:** Can capture non-linear structure and interactions without scaling. It often overfits when grown deep, leading to higher training accuracy but a wider gap with test performance. Pruning or depth limits would reduce overfitting. On Letter Recognition, single trees usually underperform ensembles; variance is high and bias is moderate.

- **K-Nearest Neighbors:** Non-parametric and sensitive to the local structure of the feature space. Scaling is important (included in the pipeline). KNN can do well when classes form compact clusters but is sensitive to irrelevant features and class imbalance. Typically shows moderate to good accuracy; its performance depends heavily on the choice of K and the distance metric. Variance can be high.

- **Naive Bayes (Gaussian):** Assumes features are conditionally independent given the class and models each class with a Gaussian. The Letter Recognition features are not truly independent, so the assumption is violated; nevertheless, Gaussian NB often gives reasonable baseline performance with very fast training. It tends to have higher bias and lower variance; AUC and MCC may lag behind more flexible models.

- **Random Forest:** Ensemble of decision trees with bagging and random feature subsets. Averages over many trees to reduce variance while keeping low bias. Usually achieves clearly better accuracy, AUC, and F1 than a single Decision Tree on this dataset. Less prone to overfitting than a single deep tree. Typically among the top performers for Letter Recognition.

- **XGBoost:** Gradient-boosted trees that sequentially correct errors of the previous model. Often reaches the best or near-best accuracy and AUC on this task. Can overfit if too many rounds or too deep trees are used; with sensible defaults it balances bias and variance well. Generally outperforms Random Forest when tuned appropriately, and both ensembles tend to outperform the single models (LR, DT, KNN, NB) on this multiclass problem.

Overall, ensembles (Random Forest and XGBoost) tend to perform better than single models due to lower variance and better use of the feature space. Logistic Regression and Naive Bayes provide stable baselines with higher bias. The Decision Tree alone is the most prone to overfitting; KNN’s performance is highly dependent on preprocessing and neighborhood size.
