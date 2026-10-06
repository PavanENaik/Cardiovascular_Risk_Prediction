# Predicting Cardiovascular Risk using Electronic Health Records

## Project Overview

This mini-project predicts the risk of cardiovascular disease using patient medical data and machine learning techniques.

The project is inspired by the Stanford CS229 reference project provided for this assignment.

## Dataset

Dataset: UCI Heart Disease Dataset

- 303 patient records
- 13 input features
- Target: Presence or absence of heart disease
- Missing values were handled during preprocessing.

## Machine Learning Models

Two models were implemented:

1. Logistic Regression
2. XGBoost

Logistic Regression was used as the baseline model, while XGBoost was used as the main model.

## Methodology

The project follows these steps:

1. Dataset collection
2. Data preprocessing
3. Missing-value handling
4. Target conversion
5. Exploratory Data Analysis
6. Train-test split
7. Model training
8. Model evaluation
9. Hyperparameter tuning
10. Final cardiovascular risk prediction

## Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- AUROC
- Confusion Matrix

## Project Structure

```text
Cardiovascular_Risk_Prediction/
│
├── data/
├── models/
├── notebooks/
├── results/
├── src/
├── .gitignore
└── README.md