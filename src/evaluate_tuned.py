import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Load tuned model
model = joblib.load("models/best_xgboost.pkl")

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_prob)

print("=" * 45)
print("TUNED XGBOOST RESULTS")
print("=" * 45)

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("AUROC    :", round(auroc, 4))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save results
results = pd.DataFrame([{
    "Model": "Tuned XGBoost",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "AUROC": auroc
}])

results.to_csv("results/tuned_xgboost_results.csv", index=False)

print("\nResults saved successfully!")