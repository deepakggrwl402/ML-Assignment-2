"""
Train 5 classification models on the German Credit Risk dataset (OpenML: credit-g,
originally UCI Statlog German Credit Data) and save fitted pipelines + metrics.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL = "class"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------
dataset = fetch_openml(name="credit-g", version=1, as_frame=True, parser="auto")
X = dataset.data.copy()
y = dataset.target.map({"good": 1, "bad": 0}).astype(int)

full_df = X.copy()
full_df[TARGET_COL] = dataset.target.values
full_df.to_csv(DATA_DIR / "german_credit_full.csv", index=False)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["category", "object"]).columns.tolist()

# ---------------------------------------------------------------------------
# 2. Train / test split (the test split is what gets shipped as test_data.csv
#    and uploaded to the Streamlit app)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

test_export = X_test.copy()
test_export[TARGET_COL] = y_test.map({1: "good", 0: "bad"}).values
test_export.to_csv(BASE_DIR / "test_data.csv", index=False)

# ---------------------------------------------------------------------------
# 3. Shared preprocessing: scale numeric, one-hot encode categorical
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

models = {
    "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(n_neighbors=15),
    "naive_bayes": GaussianNB(),
    "svm": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "knn": "KNN",
    "naive_bayes": "Naive Bayes",
    "svm": "Support Vector Machine",
}

# ---------------------------------------------------------------------------
# 4. Train each model, evaluate on the held-out test split, save pipeline
# ---------------------------------------------------------------------------
results = []
for key, clf in models.items():
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": DISPLAY_NAMES[key],
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results.append(metrics)

    joblib.dump(pipeline, MODEL_DIR / f"{key}.joblib")
    print(f"Saved {key} -> " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "Model"))

results_df = pd.DataFrame(results)
results_df.to_csv(MODEL_DIR / "metrics.csv", index=False)

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n=== Comparison Table ===")
print(results_df.round(4).to_string(index=False))

# Persist column metadata so the app can validate uploaded CSVs
schema = {
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "target_col": TARGET_COL,
    "positive_label": "good",
    "negative_label": "bad",
}
with open(MODEL_DIR / "schema.json", "w") as f:
    json.dump(schema, f, indent=2)
