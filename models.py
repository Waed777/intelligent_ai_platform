from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train_anomaly_model(df, features):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    model = IsolationForest(contamination=0.02, random_state=42)
    df["anomaly"] = model.fit_predict(X_scaled)
    df["anomaly"] = df["anomaly"].map({1: "Normal", -1: "Anomaly"})
    return df, model

def train_predictive_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model
