import streamlit as st

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Churn Prediction Dashboard", layout="wide", page_icon="📉")

# --- Imports ---
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'model.pkl'))
FEATURE_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'model_features.pkl'))
PREDICTIONS_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'predictions.csv'))

# --- LOAD MODEL & FEATURES ---
@st.cache_resource
def load_model():
    st.write(f"Loading model from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    st.write(f"Loading features from: {FEATURE_PATH}")
    features = joblib.load(FEATURE_PATH)
    return model, features

# --- LOAD PREDICTIONS ---
@st.cache_data
def load_predictions():
    st.write(f"Loading predictions from: {PREDICTIONS_PATH}")
    df = pd.read_csv(PREDICTIONS_PATH)
    df["Churn"] = (df["churn_probability"] >= 0.5).astype(int)
    return df

# --- Load Everything ---
model, model_features = load_model()
df = load_predictions()

# --- STYLES ---
st.markdown("""
    <style>
    body { background-color: #0e1117; }
    .main { color: white; }
    .css-18ni7ap { background-color: #0e1117; }
    </style>
""", unsafe_allow_html=True)

# --- TITLE ---
st.title("📉 Telco Churn Prediction Dashboard")
st.markdown("Get customer churn insights with explainability and real-time capabilities.")

# --- ROW 1: Charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔍 Churn Probability Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['churn_probability'], bins=20, kde=True, ax=ax, color='skyblue')
    st.pyplot(fig)

with col2:
    st.subheader("📊 Churn vs Retained")
    pie_data = df["Churn"].value_counts().rename(index={0: "Retained", 1: "Churn"})
    fig_pie = px.pie(values=pie_data.values, names=pie_data.index, color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_pie)

# --- ROW 2: High-Risk Customers ---
st.subheader("🚨 Top 10 High-Risk Customers")
top_churn = df.sort_values("churn_probability", ascending=False).head(10)
st.dataframe(top_churn, use_container_width=True)

# --- DOWNLOAD BUTTON ---
st.markdown("### 💾 Download Predictions")
csv_download = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", csv_download, "churn_predictions.csv", "text/csv")

# --- ROW 3: Real-time Prediction ---
st.subheader("📡 Real-Time Prediction (Simulated API)")
sample_input = st.text_input("Enter CustomerID to simulate live lookup:")
if sample_input:
    customer = df[df["customerID"] == sample_input]
    if not customer.empty:
        prob = customer["churn_probability"].values[0]
        st.success(f"Live Churn Probability: **{prob:.2f}**")
    else:
        st.error("Customer ID not found.")

# --- ROW 4: Behavioral Signal (Simulated) ---
st.subheader("🌐 External Behavior Signals (Demo Integration)")
if st.toggle("Show behavioral signal insights"):
    st.info("✅ External behavioral signal: **Low engagement detected**.\nCustomer support ticket opened 3x in last 30 days.")

# --- FOOTER ---
st.divider()
st.markdown("""
### 📱 Optimized for Mobile
This dashboard is designed to be responsive across desktop and mobile using Streamlit's layout controls.

> © Harshita Hemnani
""")
