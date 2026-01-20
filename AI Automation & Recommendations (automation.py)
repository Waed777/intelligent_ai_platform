def generate_recommendations(df):
    alerts = []
    for _, row in df.iterrows():
        if row["anomaly"] == "Anomaly" and row["risk_score"] > 0.7:
            alerts.append({
                "transaction_id": row["transaction_id"],
                "recommendation": "Increase monitoring / Auto-alert management"
            })
    return alerts
