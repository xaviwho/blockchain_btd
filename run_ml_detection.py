"""
run_ml_detection.py

This script performs ML detection on data from the MANET simulation.
Designed to run in the ML environment.
"""

import os
import sys
import json
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import time

# Add ML directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml"))

# ML configuration
MODEL_DIR = "ml/models"
RESULTS_DIR = "results"
MODEL_TYPE = "rf"
MANET_OUTPUT_DIR = "manet_output"

class NerveAgentDetector:
    """Class for nerve agent detection"""
    
    def __init__(self, model_path=None):
        """Initialize the detector with a trained model"""
        if model_path is None:
            model_path = os.path.join(MODEL_DIR, f"{MODEL_TYPE}.pkl")
        
        self.model_path = model_path
        self.load_model()
        self.load_scaler()
        
        print(f"Nerve Agent Detector initialized with {MODEL_TYPE} model")
    
    def load_model(self):
        """Load the ML model"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Loaded model from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def load_scaler(self):
        """Load the feature scaler"""
        try:
            scaler_path = os.path.join(MODEL_DIR, "concentration_scaler.pkl")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None
    
    def preprocess_reading(self, reading):
        """Preprocess a sensor reading for inference"""
        # Extract features
        features = np.array([[
            reading['concentration'],
            reading['reaction_rate'],
            reading['reaction_rate'] / 1.0  # Normalize assuming max is 1.0
        ]])
        
        # Scale features
        if self.scaler:
            features = self.scaler.transform(features)
        
        return features
    
    def detect_threat(self, reading):
        """Run inference on a sensor reading"""
        if self.model is None:
            return {"error": "Model not loaded"}
        
        # Preprocess reading
        features = self.preprocess_reading(reading)
        
        # Run inference
        start_time = time.time()
        
        # Get prediction
        prediction = self.model.predict(features)[0]
        
        # Get probability if the model supports it
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(features)[0, 1]
        else:
            probability = float(prediction)
        
        inference_time = time.time() - start_time
        
        # Create result
        result = {
            "reading_id": reading.get("id", "unknown"),
            "sensor_id": reading.get("sensor_id", "unknown"),
            "node_id": reading.get("node_id", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "threat_detected": bool(prediction),
            "threat_probability": float(probability),
            "inference_time_ms": inference_time * 1000,
            "model_type": MODEL_TYPE,
            "concentration": reading.get("concentration", 0),
            "agent": reading.get("agent", "unknown")
        }
        
        return result


class BlockchainLogger:
    """Simulates logging data to a blockchain"""
    
    def __init__(self, log_dir="blockchain_logs"):
        """Initialize the blockchain logger"""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"blockchain_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.logs = []
        print(f"Blockchain logger initialized, logging to {self.log_file}")
    
    def log_reading(self, reading, node_id):
        """Log a sensor reading to the blockchain"""
        # Create a blockchain entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "node_id": node_id,
            "reading": reading,
            "hash": self._generate_hash(reading),
            "block_number": len(self.logs) + 1
        }
        
        # Add to logs
        self.logs.append(entry)
        
        # Save periodically
        if len(self.logs) % 10 == 0:
            self._save_logs()
        
        return entry
    
    def log_detection(self, detection_result, node_id):
        """Log a detection result to the blockchain"""
        # Create a blockchain entry
        entry = {
            "timestamp": datetime.now().isoformat(),
            "node_id": node_id,
            "detection_result": detection_result,
            "hash": self._generate_hash(detection_result),
            "block_number": len(self.logs) + 1
        }
        
        # Add to logs
        self.logs.append(entry)
        
        # Save periodically
        if len(self.logs) % 10 == 0:
            self._save_logs()
        
        return entry
    
    def _generate_hash(self, data):
        """Generate a simple hash for the data (simulating blockchain)"""
        # In a real implementation, this would be a proper cryptographic hash
        import hashlib
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _save_logs(self):
        """Save logs to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)


def process_manet_data():
    """Process data from MANET simulation and perform threat detection"""
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialize ML detector
    detector = NerveAgentDetector()
    
    # Initialize blockchain logger
    blockchain = BlockchainLogger()
    
    # Find the latest MANET output file
    manet_files = glob.glob(os.path.join(MANET_OUTPUT_DIR, "manet_data_*.json"))
    if not manet_files:
        print("No MANET data files found!")
        return
    
    # Sort by modification time (newest first)
    latest_file = max(manet_files, key=os.path.getmtime)
    print(f"Using latest MANET data file: {latest_file}")
    
    # Load data from MANET simulation
    try:
        with open(latest_file, 'r') as f:
            readings = json.load(f)
        print(f"Loaded {len(readings)} readings from MANET simulation")
    except Exception as e:
        print(f"Error loading MANET data: {e}")
        return
    
    # Process readings and perform threat detection
    detection_results = []
    
    print("\nProcessing sensor data and performing threat detection...")
    
    # Process each reading
    for i, reading in enumerate(readings):
        # Get node ID from reading
        node_id = reading.get("node_id", i % 10 + 1)
        
        # Log raw reading to blockchain
        blockchain.log_reading(reading, node_id)
        
        # Perform ML detection
        detection_result = detector.detect_threat(reading)
        
        # Log detection result to blockchain
        blockchain.log_detection(detection_result, node_id)
        
        # Store result
        detection_results.append(detection_result)
        
        # Report if threat detected
        if detection_result.get('threat_detected', False):
            print(f"⚠️ THREAT DETECTED on sensor {reading.get('sensor_id')} - Agent: {reading.get('agent')} - Probability: {detection_result.get('threat_probability', 0):.4f}")
        
        # Print progress every 10 readings
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} readings")
    
    # Calculate detection statistics
    threat_count = sum(1 for r in detection_results if r.get('threat_detected', False))
    print(f"\nDetection Statistics:")
    print(f"Total readings processed: {len(detection_results)}")
    print(f"Threats detected: {threat_count} ({threat_count/len(detection_results)*100:.1f}%)")
    
    # Calculate average inference time
    inference_times = [r.get('inference_time_ms', 0) for r in detection_results]
    avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0
    print(f"Average inference time: {avg_inference_time:.2f} ms")
    
    # Save detection results
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"detection_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(detection_results, f, indent=2)
    print(f"Saved detection results to {results_file}")
    
    print("\nML detection complete!")
    print(f"Blockchain logs saved to: {blockchain.log_file}")
    print(f"Detection results saved to: {results_file}")


if __name__ == "__main__":
    process_manet_data()