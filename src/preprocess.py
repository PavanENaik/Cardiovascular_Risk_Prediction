import pandas as pd

# Load the features and target
X = pd.read_csv("data/heart_disease_features.csv")
y = pd.read_csv("data/heart_disease_target.csv")

# Combine features and target
data = X.copy()
data["target"] = y.iloc[:, 0]

# Fill missing values using the median
data["ca"] = data["ca"].fillna(data["ca"].median())
data["thal"] = data["thal"].fillna(data["thal"].median())

# Convert target to binary:
# 0 = No heart disease
# 1 = Heart disease
data["target"] = (data["target"] > 0).astype(int)

# Save the cleaned dataset
data.to_csv("data/heart_disease_cleaned.csv", index=False)

# Display results
print("Preprocessing completed!")
print("Shape:", data.shape)

print("\nTarget distribution:")
print(data["target"].value_counts())

print("\nMissing values:")
print(data.isnull().sum())