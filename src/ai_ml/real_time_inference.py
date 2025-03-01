import time
import json
import numpy as np
import pandas as pd
from web3 import Web3
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Blockchain Connection
RPC_URL = "http://43.200.53.250:8548"
ACCOUNT = "0x1D8898f42aC1330eE879e10Bf27Ca723A6B1649f"
PRIVATE_KEY = "a471850a08d06bcc47850274208275f1971c9f5888bd0a08fbc680ed9701cfda"
CONTRACT_ADDRESS = Web3.to_checksum_address("0x993e3cc1e8f252f51758f7d789f4039508c63a88")

# Contract ABI
ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "string", "name": "key", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "sensorId", "type": "uint256"},
            {"indexed": False, "internalType": "string", "name": "agent", "type": "string"}
        ],
        "name": "DataLogged",
        "type": "event"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "key", "type": "string"}
        ],
        "name": "getLog",
        "outputs": [
            {"internalType": "uint256", "name": "sensorId", "type": "uint256"},
            {"internalType": "string", "name": "agent", "type": "string"},
            {"internalType": "uint256", "name": "concentration", "type": "uint256"},
            {"internalType": "uint256", "name": "reactionRate", "type": "uint256"},
            {"internalType": "string", "name": "timestamp", "type": "string"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getLogCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "name": "logKeys",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    }
]

w3 = Web3(Web3.HTTPProvider(RPC_URL))
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

# Store last processed sensor_id to prevent duplicate processing
last_processed_id = None

def retrieve_latest_data():
    """Fetch the latest blockchain log and apply anomaly detection."""
    global last_processed_id
    total_entries = contract.functions.getLogCount().call()
    if total_entries == 0:
        print("⚠️ No data found on blockchain.")
        return None

    # Fetch the latest log entry
    key = contract.functions.logKeys(total_entries - 1).call()
    entry = contract.functions.getLog(key).call()

    log = {
        "sensor_id": entry[0],
        "agent": entry[1],
        "concentration": entry[2] / 1000,
        "reaction_rate": entry[3] / 1000,
        "timestamp": entry[4]
    }
    
    # Check if this entry has already been processed
    if last_processed_id == log["sensor_id"]:
        return None  # Ignore duplicate logs
    
    last_processed_id = log["sensor_id"]
    return log

def detect_anomalies(log_entry):
    """Apply trained AI model to detect biochemical threat anomalies."""
    model_data = pd.read_json("data/sensor_data_with_anomalies.json")
    scaler = StandardScaler()
    X_train = model_data[["concentration", "reaction_rate"]]
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(X_train_scaled)
    
    # Fix sklearn warning by using a DataFrame
    X_new = pd.DataFrame([[log_entry["concentration"], log_entry["reaction_rate"]]], columns=["concentration", "reaction_rate"])
    X_new_scaled = scaler.transform(X_new)
    anomaly_score = model.decision_function(X_new_scaled)
    anomaly_prediction = model.predict(X_new_scaled)

    log_entry["anomaly_score"] = anomaly_score[0]
    log_entry["anomaly"] = "THREAT DETECTED" if anomaly_prediction[0] == -1 else "Normal"
    return log_entry

def real_time_monitoring(interval=5):
    """Continuously monitor blockchain logs and apply AI model."""
    print("🚀 Real-Time Biochemical Threat Monitoring Started...")
    while True:
        log_entry = retrieve_latest_data()
        if log_entry:
            analyzed_log = detect_anomalies(log_entry)
            print(f"[ALERT] {analyzed_log}") if analyzed_log["anomaly"] == "THREAT DETECTED" else print("✅ All Normal")
        time.sleep(interval)  # Adjust monitoring frequency

if __name__ == "__main__":
    real_time_monitoring()
