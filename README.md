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
* **Number of features:** 12 (11 all numerical + 1 target)
* **Original target variable:** `quality` (integer score from 3 to 8)

### Target Variable Transformation

To satisfy the classification requirement, the original target variable was converted into a binary class:

* **Good Wine (1):** quality ≥ 7
* **Bad Wine (0):** quality < 7

The `quality` column was removed after creating the binary target to avoid data leakage.

---

## 3. Machine Learning Models and Evaluation  

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
| Logistic Regression | 0.89     | 0.88     | 0.69      | 0.37     | 0.48     | 0.45     |
| Decision Tree       | 0.90     | 0.80     | 0.61      | 0.67     | 0.64     | 0.59     |
| KNN                 | 0.89     | 0.82     | 0.66      | 0.41     | 0.51     | 0.47     |
| Naive Bayes         | 0.86     | 0.85     | 0.48      | 0.72     | 0.57     | 0.51     |
| Random Forest       | **0.94** | **0.95** | **0.93**  | **0.62** | **0.75** | **0.74** |
| XGBoost             | 0.94     | 0.94     | 0.87      | 0.65     | 0.74     | 0.72     |


---

## 6. Model Performance Observations

| ML Model Name           | Observation about Model Performance                                                                                                                                    |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | Achieves good overall accuracy and AUC but exhibits low recall, indicating limitations in capturing complex non-linear relationships present in the wine quality data. |
| **Decision Tree**       | Provides balanced precision and recall by effectively modeling non-linear feature interactions, though it may show slight overfitting tendencies.                      |
| **KNN**                 | Performs comparably to Logistic Regression but shows moderate recall, reflecting sensitivity to feature scaling and overlapping class distributions.                   |
| **Naive Bayes**         | Demonstrates relatively high recall but lower precision due to its strong independence assumptions, leading to over-prediction of high-quality wines.                  |
| **Random Forest**       | Achieves the best overall performance with the highest accuracy, AUC, and MCC, benefiting from ensemble learning and strong generalization.                            |
| **XGBoost**             | Delivers performance close to Random Forest by leveraging gradient boosting to capture complex non-linear patterns with high precision and recall.                     |


---

## 7. Model Implementation and Saving

All models were trained and evaluated in the **BITS Virtual Lab** using a Jupyter Notebook.
After training, the fitted models and preprocessing scaler were serialized using `joblib` and saved as `.pkl` files. These saved models are loaded directly in the Streamlit application for efficient inference without retraining.

---

## 8. Streamlit Web Application

The deployed Streamlit application includes the following features:

* **CSV dataset upload option** (test data only present in gihub test_Data folder to download and test.)
* **Model selection dropdown** for all six models
* **Display of evaluation metrics** (when ground truth is provided)
* **Confusion matrix visualization**

### Live Application Link

👉 **[https://classificationml-2025aa05683.streamlit.app/]**

---

## 9. GitHub Repository

The complete source code, trained models, and documentation are available at:

👉 **[https://github.com/anandhalakshmii/classification.ml]**

### Repository Structure

```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
|-- test_data
|   |-- wine_test_data.csv
│-- model/
│   │-- trained_model.ipynb
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
