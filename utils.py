import pandas as pd

# 1️⃣ فلترة البيانات حسب المخاطر
def filter_high_risk(df, threshold=0.7):
    return df[df["risk_score"] > threshold]

# 2️⃣ حساب نسب KPI بشكل سريع
def calculate_kpis(df):
    total_tx = len(df)
    anomalies = len(df[df["anomaly"] == "Anomaly"])
    high_risk = len(df[df["risk_score"] > 0.7])
    return {"total_tx": total_tx, "anomalies": anomalies, "high_risk": high_risk}

# 3️⃣ استخراج أعلى 10 معاملات حسب Risk Score
def top_risk_transactions(df, top_n=10):
    return df.sort_values("risk_score", ascending=False).head(top_n)

# 4️⃣ تقسيم البيانات حسب Customer Segment (مثال)
def split_by_segment(df):
    segments = df["customer_segment"].unique()
    return {seg: df[df["customer_segment"] == seg] for seg in segments}
