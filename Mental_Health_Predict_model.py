# Logistic Regression with saving model & encoders for Mental Health Dataset

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# ==== 1. Load dataset ====
df = pd.read_csv("D:\Sparsh\ML_Projects\Mental_Health_Prediction\Dataset\mental_health_data final data.csv")

# ==== 2. Handle missing values ====
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna("Unknown")  # Handle 'None' in serenity or other text columns
    else:
        df[col] = df[col].fillna(df[col].median())

# ==== 3. Label Encoding for all non-numeric columns ====
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Save encoder for future use

# ==== 4. Features & Target ====
TARGET_COL = "Mental_Health_Condition"  # Updated target column
X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# ==== 5. Train-test split ====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 6. Logistic Regression Model ====
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==== 7. Predictions ====
y_pred = model.predict(X_test)

# ==== 8. Evaluation ====
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==== 9. Save model and encoders ====

joblib.dump(model, "mental_health_model.joblib")
joblib.dump(label_encoders, "label_encoders.joblib")

print("Model and encoders saved")
