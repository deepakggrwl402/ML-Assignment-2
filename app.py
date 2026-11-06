"""
Streamlit app for the ML Assignment 2 (German Credit Risk classification).

Loads pre-trained model pipelines (preprocessing + classifier) and evaluates
them on a user-uploaded test CSV. No retraining happens in this app.
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Credit Risk Classifier", layout="wide")


@st.cache_resource
def load_schema():
    with open(MODEL_DIR / "schema.json") as f:
        return json.load(f)


@st.cache_resource
def load_model(model_file: str):
    return joblib.load(MODEL_DIR / model_file)


@st.cache_data
def load_training_metrics():
    return pd.read_csv(MODEL_DIR / "metrics.csv")


def evaluate(pipeline, X: pd.DataFrame, y_true):
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return y_pred, y_proba, metrics


def main():
    schema = load_schema()
    target_col = schema["target_col"]
    pos_label, neg_label = schema["positive_label"], schema["negative_label"]

    st.title("Credit Risk Classification — Model Evaluation Dashboard")
    st.caption(
        "BITS WILP M.Tech (AIML) — Machine Learning Assignment 2. "
        "Dataset: German Credit Risk (UCI Statlog / OpenML `credit-g`)."
    )

    st.sidebar.header("1. Upload test data")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a test CSV (must include the original feature columns and the "
        f"'{target_col}' target column with values '{pos_label}'/'{neg_label}')",
        type=["csv"],
    )
    st.sidebar.markdown("---")
    st.sidebar.header("2. Choose a model")
    selected_model_name = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

    with st.expander("Training-time comparison table (all 5 models, held-out split)"):
        st.dataframe(load_training_metrics().round(4), use_container_width=True)

    if uploaded_file is None:
        st.info("Upload the `test_data.csv` from this repo (or any compatible CSV) to see live results.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        return

    required_cols = set(schema["numeric_features"] + schema["categorical_features"] + [target_col])
    missing = required_cols - set(df.columns)
    if missing:
        st.error(f"Uploaded CSV is missing required columns: {sorted(missing)}")
        return

    X = df[schema["numeric_features"] + schema["categorical_features"]]
    y_true = df[target_col].map({pos_label: 1, neg_label: 0})
    if y_true.isna().any():
        st.error(f"The '{target_col}' column must only contain '{pos_label}' or '{neg_label}'.")
        return

    pipeline = load_model(MODEL_FILES[selected_model_name])
    y_pred, y_proba, metrics = evaluate(pipeline, X, y_true)

    st.subheader(f"Results — {selected_model_name}")
    st.caption(f"Evaluated on {len(df)} uploaded rows (model loaded as-is, no retraining).")

    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(cm, display_labels=[neg_label, pos_label]).plot(ax=ax, cmap="Blues", colorbar=False)
        st.pyplot(fig)

    with right:
        st.markdown("**Classification Report**")
        report = classification_report(
            y_true, y_pred, target_names=[neg_label, pos_label], output_dict=True
        )
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.markdown("---")
    st.subheader("Compare this model against all 5 on your uploaded data")
    if st.checkbox("Run all models on the uploaded data"):
        rows = []
        for name, file in MODEL_FILES.items():
            m = load_model(file)
            _, _, metrics_m = evaluate(m, X, y_true)
            rows.append({"Model": name, **metrics_m})
        comparison_df = pd.DataFrame(rows).round(4)
        st.dataframe(comparison_df, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        comparison_df.set_index("Model")[["Accuracy", "AUC", "F1", "MCC"]].plot(kind="bar", ax=ax2)
        plt.xticks(rotation=20, ha="right")
        st.pyplot(fig2)


if __name__ == "__main__":
    main()
