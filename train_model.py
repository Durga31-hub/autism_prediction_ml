import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("Autism_Screening_Data_Combined.csv")

# Clean column names
df.columns = df.columns.str.strip()
df.rename(columns=lambda x: x.lower(), inplace=True)

# Fix typo in column name if exists
if "jauundice" in df.columns:
    df.rename(columns={"jauundice": "jaundice"}, inplace=True)

# Drop missing values
df = df.dropna()

# Encode categorical columns
cat_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# ❗ REMOVE age and sex here
X = df.drop(["class", "age", "sex"], axis=1, errors='ignore')
y = df["class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Check imbalance
imbalance_ratio = y_train.value_counts().min() / y_train.value_counts().max()

# Apply SMOTE if needed
if imbalance_ratio < 0.7:
    sm = SMOTE(k_neighbors=3, random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

# Scaling (for LR & KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        class_weight="balanced",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

# Training & Evaluation
for name, model in models.items():
    print("\n===============================")
    print(f"Model: {name}")
    print("===============================")

    if name in ["Logistic Regression", "KNN"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importance
    if name == "Random Forest":
        feat_imp = pd.Series(
            model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)

        print("\nFeature Importance (RF):")
        print(feat_imp)

    elif name == "Logistic Regression":
        feat_imp = pd.Series(
            np.abs(model.coef_[0]),
            index=X.columns
        ).sort_values(ascending=False)

        print("\nFeature Importance (LR):")
        print(feat_imp)
