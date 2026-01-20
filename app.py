import streamlit as st
import pandas as pd
from data_generator import generate_transactions
from models import train_anomaly_model
from automation import generate_recommendations
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Big Data Risk Platform", layout="wide")
st.title("🧠 AI‑Driven Big Data Risk & Opportunity Platform")

# Generate data
df = generate_transactions(N=50000)

# ML Models
features = ["transaction_amount", "account_balance", "transaction_hour", "is_international", "merchant_risk_score"]
df, anomaly_model = train_anomaly_model(df, features)

# KPIs
total_tx = len(df)
anomalies = df[df["anomaly"] == "Anomaly"]
high_risk = df[df["risk_score"] > 0.7]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", total_tx)
col2.metric("Detected Anomalies", len(anomalies))
col3.metric("High-Risk Accounts", len(high_risk))
col4.metric("AI Recommendations Triggered", len(generate_recommendations(df)))

# Charts
st.subheader("📊 Transaction Distribution")
fig, ax = plt.subplots()
ax.hist(df["transaction_amount"], bins=50, color='skyblue')
st.pyplot(fig)

st.subheader("🚨 High Risk Anomalies")
st.dataframe(anomalies.head(20))

st.subheader("🤖 AI Recommendations")
st.dataframe(pd.DataFrame(generate_recommendations(df)))

