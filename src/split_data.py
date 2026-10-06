import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
data = pd.read_csv("data/heart_disease_cleaned.csv")

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save the split datasets
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Data split completed!")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training target distribution:")
print(y_train.value_counts())
print("\nTesting target distribution:")
print(y_test.value_counts())