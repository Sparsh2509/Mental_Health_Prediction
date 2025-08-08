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

# # -----------------------------
# # 1. Categorical Feature Counts
# categorical_cols = df.select_dtypes(include='object').columns
# for col in categorical_cols:
#     plt.figure(figsize=(6, 4))
#     sns.countplot(x=col, data=df)
#     plt.xticks(rotation=45)
#     plt.title(f'Count Plot: {col}')
#     plt.tight_layout()
#     plt.show()

# # -----------------------------
# # 2. Numerical Feature Distributions
# numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
# for col in numerical_cols:
#     plt.figure(figsize=(6, 4))
#     sns.histplot(df[col], kde=True, bins=30)
#     plt.title(f'Distribution: {col}')
#     plt.tight_layout()
#     plt.show()

# # -----------------------------
# # 3. Correlation Heatmap
# plt.figure(figsize=(10, 6))
# sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Heatmap')
# plt.tight_layout()
# plt.show()

# -----------------------------
# 4. Box Plots (Lifestyle Impact on Happiness)
# Get all categorical (object or category dtype) columns
# categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# # Loop through each and plot boxplot against Happiness Score
# for col in categorical_cols:
#     if col != 'Happiness Score':  # Avoid if 'Happiness Score' is accidentally in the list
#         plt.figure(figsize=(7, 4))
#         sns.boxplot(x=col, y='Happiness Score', data=df)
#         plt.xticks(rotation=45)
#         plt.title(f'Happiness Score by {col}')
#         plt.tight_layout()
#         plt.show()


from sklearn.preprocessing import LabelEncoder

# Make a copy to avoid changing original
df_encoded = df.copy()

# Label encode all categorical features
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
le = LabelEncoder()

for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))  # convert NaNs and non-str to str

# Now get correlation of all features (including encoded ones)
corr_matrix = df_encoded.corr()

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Numerical + Encoded Categorical Features)")
plt.tight_layout()
plt.show()


