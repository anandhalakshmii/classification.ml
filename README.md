# classification.ml
---

# 🍷 Wine Quality Classification – Machine Learning Assignment 2

---
## 1. Problem Statement

The objective of this project is to build, evaluate, and deploy multiple machine learning classification models to predict **wine quality** based on its physicochemical properties.
The task is formulated as a **binary classification problem**, where wines are classified as **Good** or **Bad** based on their quality score.

Additionally, a Streamlit web application is developed and deployed to allow interactive model selection, test data upload, and performance evaluation.

---

## 2. Dataset Description

* **Dataset Name:** Red Wine Quality Dataset
* **Source:** UCI Machine Learning Repository (via Kaggle)
* **Link:** [https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)

### Dataset Details

* **Number of instances:** 1599
* **Number of features:** 11 (all numerical)
* **Original target variable:** `quality` (integer score from 3 to 8)

### Target Variable Transformation

To satisfy the classification requirement, the original target variable was converted into a binary class:

* **Good Wine (1):** quality ≥ 7
* **Bad Wine (0):** quality < 7

The `quality` column was removed after creating the binary target to avoid data leakage.

---

## 3. Machine Learning Models and Evaluation  ✅ *(6 Marks)*

All models were trained and evaluated on the **same dataset** using an **80–20 stratified train–test split**.
Feature scaling using **StandardScaler** was applied where required.

### Models Implemented

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

---

## 4. Evaluation Metrics

For each model, the following evaluation metrics were calculated:

* Accuracy
* AUC Score
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

## 5. Model Comparison Table

| ML Model Name       | Accuracy | AUC      | Precision | Recall   | F1 Score | MCC      |
| ------------------- | -------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression | 0.89     | 0.92     | 0.88      | 0.55     | 0.68     | 0.62     |
| Decision Tree       | 0.90     | 0.88     | 0.75      | 0.75     | 0.75     | 0.71     |
| KNN                 | 0.89     | 0.91     | 0.87      | 0.56     | 0.68     | 0.62     |
| Naive Bayes         | 0.86     | 0.90     | 0.63      | 0.78     | 0.70     | 0.63     |
| Random Forest       | **0.94** | **0.96** | **0.91**  | **0.77** | **0.83** | **0.80** |
| XGBoost             | 0.94     | 0.95     | 0.92      | 0.73     | 0.81     | 0.79     |

*(Exact values may vary slightly depending on random state.)*

---

## 6. Model Performance Observations  ✅ *(3 Marks)*

| ML Model Name       | Observation about Model Performance                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------- |
| Logistic Regression | Performs well overall but has lower recall due to linear decision boundaries and class imbalance. |
| Decision Tree       | Captures non-linear patterns effectively but may slightly overfit on limited data.                |
| KNN                 | Sensitive to feature scaling and neighborhood size; performs comparably to Logistic Regression.   |
| Naive Bayes         | Shows good recall but lower precision due to independence assumptions between features.           |
| Random Forest       | Achieves the best overall performance by balancing bias and variance through ensemble learning.   |
| XGBoost             | Provides strong performance using gradient boosting, closely matching Random Forest results.      |

---

## 7. Model Implementation and Saving

All models were trained and evaluated in the **BITS Virtual Lab** using a Jupyter Notebook.
After training, the fitted models and preprocessing scaler were serialized using `joblib` and saved as `.pkl` files. These saved models are loaded directly in the Streamlit application for efficient inference without retraining.

---

## 8. Streamlit Web Application  ✅ *(4 Marks)*

The deployed Streamlit application includes the following features:

* **CSV dataset upload option** (test data only)
* **Model selection dropdown** for all six models
* **Display of evaluation metrics** (when ground truth is provided)
* **Confusion matrix visualization**

### Live Application Link

👉 **[Paste your Streamlit App Link here]**

---

## 9. GitHub Repository

The complete source code, trained models, and documentation are available at:

👉 **[Paste your GitHub Repository Link here]**

### Repository Structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│   │-- Wine_Quality_Training.ipynb
│   │-- logistic_model.pkl
│   │-- decision_tree_model.pkl
│   │-- knn_model.pkl
│   │-- naive_bayes_model.pkl
│   │-- random_forest_model.pkl
│   │-- xgboost_model.pkl
│   │-- scaler.pkl
```

---

## 10. Execution Environment

* Model training and evaluation were performed on **BITS Virtual Lab**
* A screenshot of the execution environment has been included in the final PDF submission as proof.

---

## 11. Conclusion

This project demonstrates a complete end-to-end machine learning workflow, including data preprocessing, model training, evaluation, deployment, and interactive visualization. Ensemble models such as Random Forest and XGBoost showed superior performance due to their ability to capture complex non-linear relationships in the dataset.

---
