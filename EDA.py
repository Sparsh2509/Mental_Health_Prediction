# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("D:\Sparsh\ML_Projects\Mental_Health_Prediction\Dataset\mental_health_data final data.csv")  # Update path if needed

# Basic Information
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDescriptive Statistics:\n", df.describe(include='all'))

# -----------------------------
# 1. Count Plots for Categorical Features
categorical_cols = df.select_dtypes(include='object').columns
# Exclude ID if exists
categorical_cols = [col for col in categorical_cols if col.lower() not in ['user_id', 'id']]

for col in categorical_cols:
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f'Count Plot: {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 2. Distribution Plots for Numerical Features
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numerical_cols:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Distribution Plot: {col}')
    plt.tight_layout()
    plt.show()

# -----------------------------
# 3. Correlation Heatmap (Only Numerical)
plt.figure(figsize=(10, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Box Plots for Mental Health Condition vs Lifestyle/Numerical Variables
target = 'Mental_Health_Condition'  # Update if your target has different name

# Check and plot only if target exists
if target in df.columns:
    # For numerical features
    for col in numerical_cols:
        if col != target:
            plt.figure(figsize=(7, 4))
            sns.boxplot(x=target, y=col, data=df)
            plt.title(f'{col} by {target}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # For lifestyle categorical features
    lifestyle_cats = [col for col in categorical_cols if col != target]
    for col in lifestyle_cats:
        plt.figure(figsize=(7, 4))
        sns.boxplot(x=col, y='Sleep_Hours', data=df)  # Or use another relevant numerical col
        plt.title(f'Sleep Hours by {col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
