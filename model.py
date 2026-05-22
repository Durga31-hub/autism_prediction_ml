import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# ================================
# 1. LOAD DATA
# ================================
df = pd.read_csv("Autism_Screening_Data_Combined.csv")
df.columns = df.columns.str.strip()

# Fix column names
df.rename(columns=lambda x: x.lower(), inplace=True)
if "jauundice" in df.columns:
    df.rename(columns={"jauundice": "jaundice"}, inplace=True)

# Drop missing values
df = df.dropna()

# ================================
# 2. ENCODE CATEGORICAL FEATURES
# ================================
# We need to save an encoder for EACH column to use in the backend
encoders = {}
cat_cols = df.select_dtypes(include=['object']).columns

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).str.lower())
    encoders[col] = le

# ================================
# 3. FEATURES & TARGET
# ================================
# Drop class, age, and sex as done in the testing phase
X = df.drop(["class", "age", "sex"], axis=1, errors='ignore')
y = df["class"]

# Save feature names for the backend
feature_names = list(X.columns)
joblib.dump(feature_names, 'models/feature_names.joblib')

# ================================
# 4. TRAIN-TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================================
# 5. SMOTE (HANDLE IMBALANCE)
# ================================
imbalance_ratio = y_train.value_counts().min() / y_train.value_counts().max()

if imbalance_ratio < 0.7:
    sm = SMOTE(k_neighbors=3, random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

# ================================
# 6. DEFINE & TRAIN MODEL
# ================================
print("\nTraining Gradient Boosting Model...")
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train, y_train)

# ================================
# 7. EVALUATE
# ================================
y_pred = gb_model.predict(X_test)

print("\n===============================")
print("Model: Gradient Boosting")
print("===============================")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Feature Importance
feat_imp = pd.Series(
    gb_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance (GB):")
print(feat_imp)

# ================================
# 8. SAVE MODEL & ENCODERS
# ================================
joblib.dump(gb_model, 'models/autism_model.joblib')
joblib.dump(encoders, 'models/encoders.joblib')

print("\nModel, Encoders, and Feature Names saved successfully to 'models/' directory!")
