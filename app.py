import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="AI Big Data Risk Platform", layout="wide")
st.title("🧠 AI‑Driven Big Data Risk & Opportunity Platform")

# -------- Upload User Data --------
st.subheader("📁 Upload Your Data (CSV, XLSX, XLS)")
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

# -------- Load Data --------
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.success("✅ File loaded successfully.")
    except Exception as e:
        st.error(f"❌ Error loading file: {e}")
        st.stop()
else:
    st.info("No file uploaded. Generating simulated data for demo...")
    N = 5000
    np.random.seed(42)
    df = pd.DataFrame({
        "TransactionAmount": np.random.normal(200, 50, N).clip(10, 5000),
        "Balance": np.random.normal(5000, 1500, N).clip(500, 50000),
        "Hour": np.random.randint(0, 24, N),
        "International": np.random.choice([0, 1], N, p=[0.9, 0.1]),
        "MerchantRisk": np.random.uniform(0, 1, N)
    })
    # Inject anomalies
    anomaly_idx = np.random.choice(N, int(0.02*N), replace=False)
    df.loc[anomaly_idx, "TransactionAmount"] *= 5
    df.loc[anomaly_idx, "MerchantRisk"] = np.random.uniform(0.8, 1, len(anomaly_idx))

# -------- Auto Column Mapping & Numeric Columns --------
column_mapping = {
    "TransactionAmount": "transaction_amount",
    "Balance": "account_balance",
    "Hour": "transaction_hour",
    "International": "is_international",
    "MerchantRisk": "merchant_risk_score"
}
df.rename(columns=column_mapping, inplace=True)

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if not numeric_cols:
    st.error("❌ No numeric columns found for analysis.")
    st.stop()

# -------- ML: Anomaly Detection --------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

model = IsolationForest(contamination=0.02, random_state=42)
df["anomaly"] = model.fit_predict(X_scaled)
df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})

# -------- Risk Score --------
df["risk_score"] = df[numeric_cols].mean(axis=1)  # Simple combined score

# -------- AI Recommendations --------
def generate_recommendations(df):
    recs = []
    for _, row in df.iterrows():
        if row["anomaly"] == "Anomaly" and row["risk_score"] > 0.7:
            recs.append("🔴 Investigate immediately")
        elif row["risk_score"] > 0.5:
            recs.append("🟠 Review & monitor")
        else:
            recs.append("🟢 OK / Automate process")
    return recs

recommendations = generate_recommendations(df)
df["recommendation"] = recommendations

# -------- KPIs --------
total_tx = len(df)
anomalies = df[df["anomaly"] == "Anomaly"]
high_risk = df[df["risk_score"] > 0.7]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", total_tx)
col2.metric("Detected Anomalies", len(anomalies))
col3.metric("High-Risk Accounts", len(high_risk))
col4.metric("AI Recommendations Triggered", (df["recommendation"] == "🔴 Investigate immediately").sum())

# -------- Charts --------
st.subheader("📊 Transaction Distributions (Numeric Columns)")
for col in numeric_cols:
    st.write(f"**{col}**")
    fig, ax = plt.subplots()
    ax.hist(df[col], bins=50, color='skyblue')
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    st.pyplot(fig)

# -------- High Risk Anomalies Table --------
st.subheader("🚨 High Risk Anomalies")
st.dataframe(anomalies.head(20))

# -------- AI Recommendations Table --------
st.subheader("🤖 AI Recommendations")
st.dataframe(df[["recommendation"] + numeric_cols].head(20))
