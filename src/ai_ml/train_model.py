import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load synthetic data (1000 entries) and blockchain data (404 entries)
synthetic_data_path = "data/sensor_data.json"
blockchain_data_path = "data/retrieved_sensor_data.json"

df_synthetic = pd.read_json(synthetic_data_path)
df_blockchain = pd.read_json(blockchain_data_path)

# Combine datasets for training
df = pd.concat([df_synthetic, df_blockchain], ignore_index=True)

# Convert timestamp to datetime for processing
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Select features for AI model
features = ["concentration", "reaction_rate"]
X = df[features]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train an Isolation Forest model for anomaly detection
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_scaled)

# Predict anomalies on combined data
df["anomaly_score"] = model.decision_function(X_scaled)
df["anomaly"] = model.predict(X_scaled)

# Mark anomalies (-1 = anomaly, 1 = normal)
anomalies = df[df["anomaly"] == -1]
print(f"Total Anomalies Detected: {len(anomalies)}")

# Save model and results
df.to_json("data/sensor_data_with_anomalies.json", indent=2)
print("✅ AI Model Trained on Combined Data (Synthetic + Blockchain) and Saved.")

# Visualize anomaly distribution
plt.figure(figsize=(12, 5))
plt.scatter(df["timestamp"], df["concentration"], c=df["anomaly"], cmap="coolwarm", label="Anomalies")
plt.xlabel("Time")
plt.ylabel("Concentration (ppm)")
plt.title("Anomaly Detection in Sensor Data (Blockchain + Synthetic)")
plt.legend()
plt.show()
