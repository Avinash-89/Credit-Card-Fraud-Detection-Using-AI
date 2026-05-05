import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from xgboost import XGBClassifier

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("fraud_data.csv")

# -------------------------------
# Basic checks
# -------------------------------
if "is_fraud" not in df.columns:
    raise ValueError("❌ 'is_fraud' column not found")

print("Class Distribution:\n", df["is_fraud"].value_counts())

# -------------------------------
# Feature Engineering
# -------------------------------
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

df["hour"] = df["trans_date_trans_time"].dt.hour

df["distance"] = np.sqrt(
    (df["lat"] - df["merch_lat"])**2 +
    (df["long"] - df["merch_long"])**2
)

df["is_night"] = df["hour"].apply(lambda x: 1 if x < 6 else 0)
df["high_amt"] = df["amt"].apply(lambda x: 1 if x > 500 else 0)

# -------------------------------
# Drop unnecessary columns
# -------------------------------
df = df.drop(columns=[
    "index", "trans_num", "trans_date_trans_time",
    "dob", "first", "last", "street"
], errors="ignore")

# -------------------------------
# Encode categorical variables
# -------------------------------
df = pd.get_dummies(df, drop_first=True)

# -------------------------------
# Features & target
# -------------------------------
X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,
    stratify=y if len(y.unique()) > 1 else None
)

# -------------------------------
# Scale
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# Model (XGBoost)
# -------------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.1,
    scale_pos_weight=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------------
# Evaluation with threshold tuning
# -------------------------------
if len(y_test.unique()) > 1:
    y_prob = model.predict_proba(X_test)[:, 1]

    threshold = 0.3
    y_pred = (y_prob > threshold).astype(int)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

# -------------------------------
# Feature Importance (Top 10)
# -------------------------------
importances = model.feature_importances_
features = X.columns

feat_imp = pd.Series(importances, index=features).sort_values(ascending=False)

print("\nTop Features:\n", feat_imp.head(10))

# -------------------------------
# Save everything
# -------------------------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("\n✅ Model saved successfully!")