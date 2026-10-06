import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# Load training and testing data
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# XGBoost model
xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

# Parameters to test
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1, 0.2]
}

# Grid search using AUROC
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1
)

# Train
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Test performance
y_prob = best_model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_prob)

# Save best model
joblib.dump(best_model, "models/best_xgboost.pkl")

print("Hyperparameter tuning completed!")
print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation AUROC:")
print(round(grid_search.best_score_, 4))

print("\nTest AUROC:")
print(round(auroc, 4))

print("\nBest model saved to models/best_xgboost.pkl")