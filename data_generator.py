import pandas as pd
import numpy as np

def generate_transactions(N=50000):
    np.random.seed(42)

    df = pd.DataFrame({
        "transaction_id": range(1, N+1),
        "transaction_amount": np.random.normal(200, 50, N).clip(10, 5000),
        "account_balance": np.random.normal(5000, 1500, N).clip(500, 50000),
        "transaction_hour": np.random.randint(0, 24, N),
        "is_international": np.random.choice([0, 1], N, p=[0.9, 0.1]),
        "merchant_risk_score": np.random.uniform(0, 1, N),
        "customer_segment": np.random.choice(['Retail', 'Corporate', 'VIP'], N, p=[0.7, 0.25, 0.05]),
        "product_category": np.random.choice(['Loan', 'Credit', 'Investment', 'Savings'], N)
    })

    # Inject anomalies
    anomaly_idx = np.random.choice(N, int(0.02*N), replace=False)
    df.loc[anomaly_idx, "transaction_amount"] *= 5
    df.loc[anomaly_idx, "merchant_risk_score"] = np.random.uniform(0.8, 1, len(anomaly_idx))

    # Risk Score calculation
    df["risk_score"] = (
        df["merchant_risk_score"] * 0.5 +
        (df["transaction_amount"] / df["transaction_amount"].max()) * 0.5
    )

    return df
