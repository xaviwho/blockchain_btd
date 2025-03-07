"""
inference.py

This module handles inference for nerve agent detection on MANET nodes.
It provides functions for loading models and performing real-time inference on sensor data.
"""

import os
import numpy as np
import time
import json
import threading
from collections import deque
import tensorflow as tf

# Import from other modules
from data_preparation import preprocess_node_data, load_scaler

# Constants
MODELS_DIR = "models"
SEQUENCE_LENGTH = 10
THREAT_THRESHOLD = 0.7  # Probability threshold for raising alerts
MAX_BUFFER_SIZE = 20    # Maximum number of readings to store per sensor


class NerveAgentDetector:
    """
    Class for nerve agent detection on MANET nodes
    Handles real-time inference on sensor data
    """
    
    def __init__(self, model_type="tiny", tflite=True):
        """
        Initialize the detector
        
        Args:
            model_type (str): Type of model to load ('tiny', 'cnn', 'lstm', 'hybrid')
            tflite (bool): Whether to use TFLite model (recommended for edge devices)
        """
        self.model_type = model_type
        self.tflite = tflite
        self.sensor_buffers = {}  # Buffer to store recent readings for each sensor
        self.ready = False
        
        # Load model and scaler
        self.load_model()
        self.load_scaler()
        
        self.ready = True
        print(f"Nerve Agent Detector initialized with {model_type} model")
    
    def load_model(self):
        """Load the ML model"""
        try:
            if self.tflite:
                # Load TFLite model
                model_path = os.path.join(MODELS_DIR, f"{self.model_type}.tflite")
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                
                # Get input and output details
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                
                print(f"Loaded TFLite model from {model_path}")
            else:
                # Load Keras model
                model_path = os.path.join(MODELS_DIR, f"{self.model_type}.h5")
                self.model = tf.keras.models.load_model(model_path)
                print(f"Loaded Keras model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.ready = False
    
    def load_scaler(self):
        """Load the scaler for feature normalization"""
        try:
            scaler_path = os.path.join(MODELS_DIR, "concentration_scaler.pkl")
            self.scaler = load_scaler(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.ready = False
    
    def add_reading(self, sensor_id, reading):
        """
        Add a new sensor reading to the buffer
        
        Args:
            sensor_id: ID of the sensor
            reading: Sensor reading data
            
        Returns:
            bool: True if the reading was added successfully
        """
        if not self.ready:
            print("Detector not ready")
            return False
        
        # Initialize buffer for this sensor if it doesn't exist
        if sensor_id not in self.sensor_buffers:
            self.sensor_buffers[sensor_id] = deque(maxlen=MAX_BUFFER_SIZE)
        
        # Add reading to buffer
        self.sensor_buffers[sensor_id].append(reading)
        
        return True
    
    def detect(self, sensor_id):
        """
        Run inference on the latest readings for a sensor
        
        Args:
            sensor_id: ID of the sensor to analyze
            
        Returns:
            dict: Results including threat probability and detection status
        """
        if not self.ready:
            return {"error": "Detector not ready"}
        
        if sensor_id not in self.sensor_buffers:
            return {"error": f"No readings for sensor {sensor_id}"}
            
        if len(self.sensor_buffers[sensor_id]) < SEQUENCE_LENGTH:
            return {"error": f"Not enough readings for sensor {sensor_id}. Need {SEQUENCE_LENGTH}, have {len(self.sensor_buffers[sensor_id])}"}
        
        # Preprocess data
        readings = list(self.sensor_buffers[sensor_id])
        sequence = preprocess_node_data(readings, SEQUENCE_LENGTH, self.scaler)
        
        # Run inference
        start_time = time.time()
        
        if self.tflite:
            # TFLite inference
            input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            threat_probability = self.interpreter.get_tensor(self.output_details[0]['index'])[0][0]
        else:
            # Keras inference
            input_data = np.expand_dims(sequence, axis=0)
            threat_probability = self.model.predict(input_data, verbose=0)[0][0]
        
        inference_time = time.time() - start_time
        
        # Determine if threat is detected
        threat_detected = threat_probability > THREAT_THRESHOLD
        
        # Create result
        result = {
            "sensor_id": sensor_id,
            "timestamp": time.time(),
            "threat_probability": float(threat_probability),
            "threat_detected": bool(threat_detected),
            "inference_time": inference_time,
            "model_type": self.model_type
        }
        
        return result


class SensorMonitor:
    """
    Class that monitors sensors and runs periodic threat detection
    """
    
    def __init__(self, detector, log_dir="logs"):
        """
        Initialize the sensor monitor
        
        Args:
            detector: NerveAgentDetector instance
            log_dir: Directory to save detection logs
        """
        self.detector = detector
        self.log_dir = log_dir
        self.monitoring = False
        self.monitoring_thread = None
        self.detection_interval = 5  # seconds
        self.detection_logs = []
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if self.monitoring:
            print("Already monitoring")
            return
        
        self.monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        print("Sensor monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
        print("Sensor monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread"""
        while self.monitoring:
            # Check all sensors
            for sensor_id in list(self.detector.sensor_buffers.keys()):
                try:
                    result = self.detector.detect(sensor_id)
                    if "error" not in result:
                        self.detection_logs.append(result)
                        
                        # If threat detected, log it prominently
                        if result["threat_detected"]:
                            print(f"⚠️ THREAT DETECTED on sensor {sensor_id}: {result['threat_probability']:.4f}")
                except Exception as e:
                    print(f"Error detecting threats for sensor {sensor_id}: {e}")
            
            # Save logs periodically
            if len(self.detection_logs) >= 10:
                self.save_logs()
            
            # Sleep until next detection cycle
            time.sleep(self.detection_interval)
    
    def save_logs(self):
        """Save detection logs to file"""
        if not self.detection_logs:
            return
        
        try:
            # Create filename with timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.log_dir, f"detection_log_{timestamp}.json")
            
            # Save logs
            with open(filename, 'w') as f:
                json.dump(self.detection_logs, f, indent=2)
            
            print(f"Saved {len(self.detection_logs)} detection logs to {filename}")
            
            # Clear logs
            self.detection_logs = []
        except Exception as e:
            print(f"Error saving detection logs: {e}")


def run_inference_on_file(file_path, model_type="tiny", output_path=None):
    """
    Run inference on a file containing sensor readings
    Useful for offline testing
    
    Args:
        file_path: Path to JSON file with sensor readings
        model_type: Type of model to use
        output_path: Path to save results (optional)
        
    Returns:
        list: Detection results
    """
    # Load data
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Create detector
    detector = NerveAgentDetector(model_type=model_type)
    
    # Group readings by sensor
    sensor_readings = {}
    for reading in data:
        sensor_id = reading["sensor_id"]
        if sensor_id not in sensor_readings:
            sensor_readings[sensor_id] = []
        sensor_readings[sensor_id].append(reading)
    
    # Process each sensor
    results = []
    for sensor_id, readings in sensor_readings.items():
        print(f"Processing {len(readings)} readings for sensor {sensor_id}")
        
        # Add readings to detector
        for reading in readings:
            detector.add_reading(sensor_id, reading)
        
        # Run detection at regular intervals
        for i in range(0, len(readings), SEQUENCE_LENGTH):
            if i >= SEQUENCE_LENGTH:
                result = detector.detect(sensor_id)
                if "error" not in result:
                    results.append(result)
    
    # Save results if output path provided
    if output_path and results:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved {len(results)} results to {output_path}")
    
    return results


if __name__ == "__main__":
    # Example: Run inference on a file
    results = run_inference_on_file(
        "./data/sensor_data.json",
        model_type="tiny",
        output_path="results/file_inference_results.json"
    )
    
    # Count threats
    threats = [r for r in results if r["threat_detected"]]
    print(f"Detected {len(threats)} threats in {len(results)} samples")
    
    # Example: Create a detector and monitor
    detector = NerveAgentDetector(model_type="tiny")
    monitor = SensorMonitor(detector)
    
    # Add some sample readings
    with open("./data/sensor_data.json", 'r') as f:
        sample_data = json.load(f)
    
    for i in range(20):
        reading = sample_data[i]
        detector.add_reading(reading["sensor_id"], reading)
    
    # Run monitoring for a short time
    monitor.start_monitoring()
    time.sleep(10)
    monitor.stop_monitoring()