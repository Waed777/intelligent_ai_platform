import streamlit as st
import pandas as pd
from data_generator import generate_transactions
from models import train_anomaly_model
from automation import generate_recommendations
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Big Data Risk Platform", layout="wide")
st.title("🧠 AI‑Driven Big Data Risk & Opportunity Platform")

# -------- Upload User Data --------
st.subheader("📁 Upload Your Data (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())
else:
    # Generate simulated data if no upload
    df = generate_transactions(N=50000)
    st.info("No file uploaded. Using simulated data for demo purposes.")

# -------- ML Models --------
features = ["transaction_amount", "account_balance", "transaction_hour", "is_international", "merchant_risk_score"]
df, anomaly_model = train_anomaly_model(df, features)

# -------- AI Recommendations --------
recommendations = generate_recommendations(df)

# -------- KPIs --------
total_tx = len(df)
anomalies = df[df["anomaly"] == "Anomaly"]
high_risk = df[df["risk_score"] > 0.7]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", total_tx)
col2.metric("Detected Anomalies", len(anomalies))
col3.metric("High-Risk Accounts", len(high_risk))
col4.metric("AI Recommendations Triggered", len(recommendations))

# -------- Charts --------
st.subheader("📊 Transaction Distribution")
fig, ax = plt.subplots()
ax.hist(df["transaction_amount"], bins=50, color='skyblue')
ax.set_xlabel("Transaction Amount")
ax.set_ylabel("Count")
st.pyplot(fig)

# -------- High Risk Anomalies Table --------
st.subheader("🚨 High Risk Anomalies")
st.dataframe(anomalies.head(20))

# -------- AI Recommendations Table --------
st.subheader("🤖 AI Recommendations")
st.dataframe(pd.DataFrame(recommendations))

st.dataframe(pd.DataFrame(generate_recommendations(df)))

