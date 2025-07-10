import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.set_page_config(page_title="📊 Student Score Predictor", layout="wide")
st.title("🎓 Student Score Predictor with ML")

st.markdown("Upload a CSV file with the following columns: `hours_studied`, `attendance_rate`, `previous_score`, and `score`.")

uploaded_file = st.file_uploader("Upload your student data CSV", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data.columns = data.columns.str.strip()

    required_cols = {"hours_studied", "attendance_rate", "previous_score", "score"}
    if not required_cols.issubset(data.columns):
        st.error(f"❌ Uploaded file must contain these columns: {required_cols}")
    else:
        X = data.drop("score", axis=1)
        y = data["score"]

        # Load best model
        with open("best_model_pipeline.pkl", "rb") as f:
            model = pickle.load(f)

        predictions = model.predict(X)

        st.subheader("📈 Predictions")
        result_df = data.copy()
        result_df["Predicted Score"] = predictions
        st.dataframe(result_df)

        # 📉 Actual vs Predicted Plot
        st.subheader("📉 Actual vs Predicted Scores")
        fig, ax = plt.subplots()
        ax.scatter(data["score"], predictions, color='teal')
        ax.plot([data["score"].min(), data["score"].max()], [data["score"].min(), data["score"].max()], 'r--')
        ax.set_xlabel("Actual Score")
        ax.set_ylabel("Predicted Score")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # 📊 Correlation Heatmap
        st.subheader("📊 Feature Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

        # 📋 Feature Importances (only if model has it)
        if hasattr(model.named_steps['model'], 'feature_importances_'):
            st.subheader("🔍 Feature Importances")
            importances = model.named_steps['model'].feature_importances_
            features = X.columns
            fig, ax = plt.subplots()
            sns.barplot(x=importances, y=features, ax=ax, palette="coolwarm")
            ax.set_title("Feature Importances")
            st.pyplot(fig)

# Display saved visuals if no file uploaded
else:
    st.warning("📂 Upload a dataset to begin model predictions.")
    if st.button("Show Heatmap & Comparison Charts"):
        st.image("correlation_heatmap.png", caption="Correlation Heatmap", use_column_width=True)
        st.image("model_comparison_visuals.png", caption="Model Comparison", use_column_width=True)
