import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
data = pd.read_csv("data/heart_disease_cleaned.csv")

# Basic information
print("Dataset Shape:", data.shape)
print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# Target distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=data)
plt.title("Heart Disease Distribution")
plt.xlabel("Heart Disease (0 = No, 1 = Yes)")
plt.ylabel("Number of Patients")
plt.savefig("results/target_distribution.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("results/correlation_heatmap.png")
plt.show()

print("\nEDA completed!")