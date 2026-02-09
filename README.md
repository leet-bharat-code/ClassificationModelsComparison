# Classification Models

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
| Logistic Regression | 0.7712 | 0.9808 | 0.7708 | 0.7699 | 0.7695 | 0.7621 |
| Decision Tree | 0.8742 | 0.9346 | 0.8751 | 0.8742 | 0.8743 | 0.8692 | 
| K-Nearest Neighbors | 0.9400 | 0.9956 | 0.9413 | 0.9397 | 0.9400 | 0.9376 |
| Naive Bayes | 0.6506 | 0.9576 | 0.6614 | 0.6494 | 0.6456 | 0.6374 |
| Random Forest | 0.9680 | 0.9996 | 0.9682 | 0.9677 | 0.9678 | 0.9667 |
| XGBoost | 0.9654 | 0.9997 | 0.9655 | 0.9652 | 0.9652 | 0.9640 |

---

## 4. Observations

- **Logistic Regression:**  
  Logistic Regression shows lower performance compared to tree-based and ensemble models. While the AUC is relatively high, the overall accuracy, F1-score, and MCC are noticeably lower. This indicates that the linear decision boundary is insufficient to capture the complex, non-linear patterns present in the Letter Recognition dataset. The model exhibits higher bias and tends to underfit, making it more suitable as a baseline rather than a strong classifier for this task.

- **Decision Tree:**  
  The Decision Tree performs better than Logistic Regression and Naive Bayes but lags behind ensemble models. It is able to capture non-linear relationships in the data, which improves accuracy and F1-score. However, as a single tree, it is prone to overfitting and has higher variance. This limits its generalization performance on the test set, explaining the gap compared to Random Forest and XGBoost.

- **K-Nearest Neighbors:**  
  KNN achieves strong performance with high accuracy, F1-score, and AUC. This suggests that the feature space contains meaningful local neighborhoods that help classify letters effectively. However, KNN still performs slightly worse than ensemble methods and can be computationally expensive during inference. Its performance is sensitive to the choice of K and distance metric, which affects stability.

- **Naive Bayes (Gaussian):**  
  Naive Bayes shows the weakest performance among all evaluated models. Although the AUC is reasonably high, the low accuracy, F1-score, and MCC indicate poor overall classification quality. This is expected because the conditional independence assumption of Naive Bayes does not hold for the Letter Recognition features, which are correlated. As a result, Naive Bayes serves mainly as a fast and simple baseline model.

- **Random Forest:**  
  Random Forest delivers the best overall performance across almost all metrics, including accuracy, precision, recall, F1-score, and MCC. By combining multiple decision trees and averaging their predictions, the model effectively reduces overfitting while maintaining strong predictive power. Its near-perfect AUC reflects excellent class separability. This makes Random Forest the most reliable and well-balanced model for this dataset.

- **XGBoost:**  
  XGBoost also achieves excellent performance, closely matching Random Forest across all metrics. Its extremely high AUC indicates very strong discrimination between classes. While slightly below Random Forest in some metrics, XGBoost remains highly competitive and demonstrates the strength of boosting-based ensembles. With careful tuning, it can match or even surpass Random Forest, making it another strong candidate for deployment.

**Overall Conclusion:**  
Ensemble models, particularly **Random Forest** and **XGBoost**, outperform single and linear models on the Letter Recognition task due to their ability to capture complex feature interactions and reduce variance. **KNN** performs well but is less scalable, while **Decision Tree**, **Logistic Regression**, and **Naive Bayes** show progressively lower performance due to overfitting, underfitting, or restrictive assumptions. 
