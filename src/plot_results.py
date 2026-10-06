import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

# Load test data
X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()

# Load models
logistic_model = joblib.load("models/logistic_regression.pkl")
xgb_model = joblib.load("models/xgboost.pkl")

models = {
    "Logistic Regression": logistic_model,
    "XGBoost": xgb_model
}

plt.figure(figsize=(8, 6))

for name, model in models.items():

    # Get probability predictions
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUROC = {roc_auc:.3f})"
    )

# Random classifier line
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Cardiovascular Risk Prediction")
plt.legend()
plt.grid()

plt.savefig("results/roc_curve.png")
plt.show()

print("ROC curve saved successfully!")