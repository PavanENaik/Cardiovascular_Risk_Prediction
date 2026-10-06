import pandas as pd
import joblib

# Load the best trained model
model = joblib.load("models/best_xgboost.pkl")

# Patient information
patient = pd.DataFrame([{
    "age": 55,
    "sex": 1,
    "cp": 1,
    "trestbps": 140,
    "chol": 250,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 1,
    "ca": 0,
    "thal": 2
}])

# Predict
prediction = model.predict(patient)[0]
probability = model.predict_proba(patient)[0][1]

print("=" * 45)
print("CARDIOVASCULAR RISK PREDICTION")
print("=" * 45)

print("\nRisk probability:", round(probability * 100, 2), "%")

if prediction == 1:
    print("Prediction: HIGHER RISK OF HEART DISEASE")
else:
    print("Prediction: LOWER RISK OF HEART DISEASE")