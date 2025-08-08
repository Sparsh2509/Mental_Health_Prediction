# Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Dataset
df = pd.read_csv("D:\Sparsh\ML_Projects\Mental_Health_Prediction\Dataset\Mental_Health_Lifestyle_Dataset.csv")  # Update path if needed

# Basic Info
print("Shape:", df.shape)
print("\nColumn Names:\n", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df.describe())

# -----------------------------
# 1. Categorical Feature Counts
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df)
    plt.xticks(rotation=45)
    plt.title(f'Count Plot: {col}')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 2. Numerical Feature Distributions
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution: {col}')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 3. Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Box Plots (Lifestyle Impact on Happiness)
lifestyle_cols = ['Smoking Status', 'Diet Type', 'Work Type', 'BMI Category', 'Alcohol Consumption']
for col in lifestyle_cols:
    if col in df.columns:
        plt.figure(figsize=(7, 4))
        sns.boxplot(x=col, y='Happiness Score', data=df)
        plt.xticks(rotation=45)
        plt.title(f'Happiness Score by {col}')
        plt.tight_layout()
        plt.show()
