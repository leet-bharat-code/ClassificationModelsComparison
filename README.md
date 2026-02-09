# Classification Models — Academic Submission

## Required workflow

1. `pip install -r requirements.txt`
2. `python -m model.train_models` — trains all models and computes metrics (writes `data/evaluation_results.csv`; no .joblib committed)
3. `python scripts/print_eval_table.py` — prints the evaluation table to stdout; copy into Section 3 below
4. `streamlit run app.py`
5. Deploy on Streamlit Community Cloud (training runs automatically if needed; no artifacts required)

**Dataset:** If `data/letter-recognition.data` does not exist, the first run downloads it from UCI and saves it to `data/`. For offline use, place the file in `data/` before running.

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

Dataset handling: if the file exists in `data/`, it is loaded from there; otherwise it is downloaded from the UCI URL and saved to `data/`. Train/test split uses a fixed `random_state=42` (75% train / 25% test). Preprocessing uses `StandardScaler` inside sklearn Pipelines.

---

## 3. Models Used and Evaluation

All models are trained on the same preprocessed data and evaluated on the same test set. Metrics (Accuracy, AUC, Precision, Recall, F1, MCC) are computed with `sklearn.metrics` and stored in a pandas DataFrame and in `data/evaluation_results.csv`. Table below: paste output of `python scripts/print_eval_table.py`.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|-----|------------|--------|-----|-----|
| Logistic Regression | (paste from print_eval_table.py) |
| Decision Tree | (paste from print_eval_table.py) |
| K-Nearest Neighbors | (paste from print_eval_table.py) |
| Naive Bayes | (paste from print_eval_table.py) |
| Random Forest | (paste from print_eval_table.py) |
| XGBoost | (paste from print_eval_table.py) |

---

## 4. Observations

- **Logistic Regression:** A linear model that learns decision boundaries in the scaled feature space. It typically achieves moderate accuracy on Letter Recognition because the problem is not linearly separable; some letter pairs (e.g. O vs Q) are similar in the given features. It is stable and has low variance but can underfit, leading to lower AUC and recall on harder classes. Bias-variance: higher bias, lower variance.

- **Decision Tree:** Can capture non-linear structure and interactions without scaling. It often overfits when grown deep, leading to higher training accuracy but a wider gap with test performance. Pruning or depth limits would reduce overfitting. On Letter Recognition, single trees usually underperform ensembles; variance is high and bias is moderate.

- **K-Nearest Neighbors:** Non-parametric and sensitive to the local structure of the feature space. Scaling is important (included in the pipeline). KNN can do well when classes form compact clusters but is sensitive to irrelevant features and class imbalance. Typically shows moderate to good accuracy; its performance depends heavily on the choice of K and the distance metric. Variance can be high.

- **Naive Bayes (Gaussian):** Assumes features are conditionally independent given the class and models each class with a Gaussian. The Letter Recognition features are not truly independent, so the assumption is violated; nevertheless, Gaussian NB often gives reasonable baseline performance with very fast training. It tends to have higher bias and lower variance; AUC and MCC may lag behind more flexible models.

- **Random Forest:** Ensemble of decision trees with bagging and random feature subsets. Averages over many trees to reduce variance while keeping low bias. Usually achieves clearly better accuracy, AUC, and F1 than a single Decision Tree on this dataset. Less prone to overfitting than a single deep tree. Typically among the top performers for Letter Recognition.

- **XGBoost:** Gradient-boosted trees that sequentially correct errors of the previous model. Often reaches the best or near-best accuracy and AUC on this task. Can overfit if too many rounds or too deep trees are used; with sensible defaults it balances bias and variance well. Generally outperforms Random Forest when tuned appropriately, and both ensembles tend to outperform the single models (LR, DT, KNN, NB) on this multiclass problem.

Overall, ensembles (Random Forest and XGBoost) tend to perform better than single models due to lower variance and better use of the feature space. Logistic Regression and Naive Bayes provide stable baselines with higher bias. The Decision Tree alone is the most prone to overfitting; KNN’s performance is highly dependent on preprocessing and neighborhood size.
