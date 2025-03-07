"""
manet_with_ml.py

This script integrates the MANET simulation with ML-based nerve agent detection
and blockchain logging for a complete tactical edge detection system.
"""

# Add the ML directory to the Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml"))

# Import CORE API
from core.api.grpc import client
from core.api.grpc.wrappers import SessionState, Node, Position, Interface, Link

# Other imports
import time
import json
import pickle
import threading
import numpy as np
import pandas as pd
from datetime import datetime
import random

# Paths and directories
MODEL_DIR = "ml/models"
RESULTS_DIR = "results"
DATA_PATH = "data/sensor_data.json"
PCAP_DIR = "manet_pcaps"
XML_PATH = "manet_session.xml"

# ML and detection parameters
MODEL_TYPE = "rf"  # Options: rf, gb, svm, knn, mlp
THREAT_THRESHOLD = 0.7
DETECTION_INTERVAL = 5  # seconds


class NerveAgentDetector:
    """Class for nerve agent detection on MANET nodes"""
    
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


def create_manet_with_ml(pcap_dir=PCAP_DIR, xml_path=XML_PATH):
    """
    Create a MANET simulation with ML-based nerve agent detection
    
    Args:
        pcap_dir: Directory to save pcap files
        xml_path: Path to save the session XML
    """
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Set default paths if not provided
    if pcap_dir is None:
        pcap_dir = PCAP_DIR
    else:
        os.makedirs(pcap_dir, exist_ok=True)
        
    if xml_path is None:
        xml_path = XML_PATH
    
    # Ensure parent directory for xml_path exists
    os.makedirs(os.path.dirname(os.path.abspath(xml_path)), exist_ok=True)
    
    print(f"PCAP files will be saved to: {pcap_dir}")
    print(f"Session XML will be saved to: {xml_path}")
    
    # Initialize ML detector
    detector = NerveAgentDetector()
    
    # Initialize blockchain logger
    blockchain = BlockchainLogger()
    
    # Connect to core-daemon gRPC server
    core = client.CoreGrpcClient()
    core.connect()  # Connect to default localhost:50051
    
    # Create a new session
    session = core.create_session()
    print(f"Created new session with ID: {session.id}")
    
    # Set session state to CONFIGURATION
    session.state = SessionState.CONFIGURATION
    print("Set session state to CONFIGURATION")
    
    # Create 10 stations and 1 AP
    stations = []
    for i in range(10):
        # Create a Node object
        node = Node(
            name=f'sta{i}',
            model='wlan',
            position=Position(x=i * 50, y=100)  # Spread out nodes for better visibility
        )
        # Add node to the session
        node_id = core.add_node(session_id=session.id, node=node)
        stations.append(node_id)
        print(f"Created station {i} with node_id {node_id}")
        
        # Configure interface using node_command
        try:
            core.node_command(
                session_id=session.id,
                node_id=node_id,
                command=f'ifconfig wlan0 10.0.0.{i+1}/24 up'
            )
            print(f"Configured interface for station {i}")
        except Exception as e:
            print(f"Error configuring interface for station {i}: {e}")
    
    # Create AP node
    ap_node = Node(
        name='ap1',
        model='wlan_ap',
        position=Position(x=250, y=200)  # Position AP in the middle, slightly below
    )
    # Add AP node to the session
    ap_id = core.add_node(session_id=session.id, node=ap_node)
    print(f"Created AP with node_id {ap_id}")
    
    # Configure AP interface
    try:
        core.node_command(
            session_id=session.id,
            node_id=ap_id,
            command='ifconfig wlan0 10.0.0.254/24 up'
        )
        print("Configured interface for AP")
    except Exception as e:
        print(f"Error configuring interface for AP: {e}")

    # Configure WiFi using set_wlan_config with correct parameters
    for node_id in stations + [ap_id]:
        try:
            # From the debug output, we know set_wlan_config takes (session_id, node_id, config)
            core.set_wlan_config(
                session_id=session.id,
                node_id=node_id,
                config={"ssid": "manet", "channel": "1"}
            )
            print(f"Configured WiFi for node {node_id}")
        except Exception as e:
            print(f"Error configuring WiFi for node {node_id}: {e}")

    # Link stations to AP with OVS switches (star topology)
    switches = []
    for i, sta_id in enumerate(stations):
        # Create OVS switch
        ovs_switch = Node(
            name=f'switch_{i}',
            model='ovs',
            position=Position(x=i * 50, y=150)  # Position switches between stations and AP
        )
        ovs_switch_id = core.add_node(session_id=session.id, node=ovs_switch)
        switches.append(ovs_switch_id)
        print(f"Created switch {i} with node_id {ovs_switch_id}")
        
        # Create Interface objects with unique IDs
        # For station to switch link
        sta_iface = Interface(id=i+1, name=f"eth{i+1}")
        switch_iface1 = Interface(id=i+1, name=f"eth{i+1}")
        
        # For switch to AP link
        switch_iface2 = Interface(id=i+10, name=f"eth{i+10}")
        ap_iface = Interface(id=i+1, name=f"eth{i+1}")
        
        # Create Link objects with interfaces
        link1 = Link(
            node1_id=sta_id, 
            node2_id=ovs_switch_id,
            iface1=sta_iface,
            iface2=switch_iface1
        )
        
        link2 = Link(
            node1_id=ovs_switch_id, 
            node2_id=ap_id,
            iface1=switch_iface2,
            iface2=ap_iface
        )
        
        # Add links using the correct parameters
        try:
            core.add_link(session_id=session.id, link=link1)
            print(f"Created link from station {i} to switch {i}")
        except Exception as e:
            print(f"Error creating link from station {i} to switch {i}: {e}")
        
        try:
            core.add_link(session_id=session.id, link=link2)
            print(f"Created link from switch {i} to AP")
        except Exception as e:
            print(f"Error creating link from switch {i} to AP: {e}")

    # Set 20% packet loss on wireless links
    for i, sta_id in enumerate(stations):
        if i < len(switches):
            switch_id = switches[i]
            try:
                # Create a new link with loss option
                options = {"loss": 20.0}
                loss_link = Link(
                    node1_id=sta_id, 
                    node2_id=switch_id, 
                    options=options,
                    iface1=Interface(id=i+1, name=f"eth{i+1}"),
                    iface2=Interface(id=i+1, name=f"eth{i+1}")
                )
                # Edit the link
                core.edit_link(session_id=session.id, link=loss_link)
                print(f"Set 20% packet loss on link between station {i} and switch {i}")
            except Exception as e:
                print(f"Error setting packet loss: {e}")

    # Enable OLSR routing on all nodes
    for node_id in stations + [ap_id]:
        try:
            core.node_command(
                session_id=session.id,
                node_id=node_id,
                command='olsrd -d 0 &'
            )
            print(f"Started OLSR on node {node_id}")
        except Exception as e:
            print(f"Error starting OLSR on node {node_id}: {e}")

    # Save session to XML
    try:
        # Try with positional arguments (session_id, file_path)
        core.save_xml(session.id, xml_path)
        print(f"Saved session XML to {xml_path}")
    except Exception as e:
        print(f"Error saving session XML: {e}")
        try:
            # Try with keyword arguments
            core.save_xml(session_id=session.id, file_path=xml_path)
            print(f"Saved session XML to {xml_path} (using keyword arguments)")
        except Exception as e2:
            print(f"Could not save session XML: {e2}")
    
    # Try to start the session
    print("\nAttempting to start session...")
    try:
        # Try different approaches to start the session
        try:
            # Try with positional argument
            core.start_session(session.id)
            print("Session started successfully using start_session!")
        except Exception as e1:
            print(f"Error with start_session (positional): {e1}")
            try:
                # Try direct state modification
                session.state = SessionState.RUNTIME
                print("Set session state to RUNTIME directly")
            except Exception as e2:
                print(f"Error setting session state: {e2}")
                # As a last resort, print instructions for manual start
                print("\n*** MANUAL STEPS REQUIRED ***")
                print(f"Please start session {session.id} manually in the CORE GUI.")
                print("1. Open CORE GUI (if not already open)")
                print("2. Open Sessions dialog (Ctrl+Shift+N)")
                print(f"3. Connect to session {session.id}")
                print("4. From Session menu, select Start")
                print("***************************\n")
    except Exception as e:
        print(f"All attempts to start session failed: {e}")
    
    print("MANET created and started. Testing connectivity...")

    # Set default routes for stations (to AP)
    for sta_id in stations:
        core.node_command(session_id=session.id, node_id=sta_id, command='route add default gw 10.0.0.254 wlan0')
        print(f"Set default route for station {sta_id}")

    # Load sensor data
    try:
        with open(DATA_PATH, 'r') as f:
            sensor_data = json.load(f)
        print(f"Loaded {len(sensor_data)} sensor readings from {DATA_PATH}")
    except Exception as e:
        print(f"Error loading sensor data: {e}")
        sensor_data = []
    
    # Start detection monitoring on each station
    detection_results = []
    
    # Simulate PureChain data transmission with ML detection
    try:
        # We'll use fewer entries for demonstration
        for i, reading in enumerate(sensor_data[:100]):
            # Select a random station as the sensor
            sender_id = stations[i % 10]
            
            # Log the raw reading to blockchain
            blockchain.log_reading(reading, sender_id)
            
            # Run ML detection on the reading
            detection_result = detector.detect_threat(reading)
            
            # Log detection result to blockchain
            blockchain.log_detection(detection_result, sender_id)
            
            # Store result for reporting
            detection_results.append(detection_result)
            
            # Report if threat detected
            if detection_result.get('threat_detected', False):
                print(f"⚠️ THREAT DETECTED on sensor {reading.get('sensor_id')} - Agent: {reading.get('agent')} - Probability: {detection_result.get('threat_probability', 0):.4f}")
            
            # Create readable format for transmission
            data_str = f"Sensor-{reading['sensor_id']}-Agent-{reading['agent']}-Conc-{reading['concentration']}"
            
            # Transmit the data through the MANET
            core.node_command(session_id=session.id, node_id=sender_id, command=f'echo "{data_str}" | nc -u 10.0.0.254 12345')
            
            # Add some delay between readings
            time.sleep(0.1)
            
            # Print progress every 10 transmissions
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1} readings")
    
    except Exception as e:
        print(f"Error simulating PureChain data: {e}")

    # Start tcpdump on stations to capture traffic
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pcap_files = []
    
    for i, sta_id in enumerate(stations[:3]):
        pcap_path = os.path.join(pcap_dir, f"wifi_traffic_sta{i+1}_{timestamp}.pcap")
        pcap_files.append(pcap_path)
        
        core.node_command(
            session_id=session.id,
            node_id=sta_id,
            command=f'tcpdump -i wlan0 -w {pcap_path} &'
        )
        print(f"Started tcpdump on station {i+1}, saving to {pcap_path}")

    print("\nSimulated PureChain data transmission with ML detection. Capturing traffic...")
    
    # Run for 30 seconds to capture ongoing traffic
    print("Running capture for 30 seconds...")
    time.sleep(30)

    # Stop tcpdump
    for sta_id in stations[:3]:
        core.node_command(session_id=session.id, node_id=sta_id, command='pkill -9 tcpdump')
    print("Stopped tcpdump on all stations")

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
    with open(os.path.join(RESULTS_DIR, f"detection_results_{timestamp}.json"), 'w') as f:
        json.dump(detection_results, f, indent=2)
    print(f"Saved detection results to {os.path.join(RESULTS_DIR, f'detection_results_{timestamp}.json')}")
    
    # Print summary
    print("\nMANET with ML Simulation Summary:")
    print("=======================")
    print(f"Session ID: {session.id}")
    print(f"Total Nodes: {len(stations) + len(switches) + 1}")
    print("Captured traffic files:")
    for pcap_file in pcap_files:
        print(f" - {pcap_file}")
    print(f"Session XML saved to: {xml_path}")
    print(f"Detection results saved to: {os.path.join(RESULTS_DIR, f'detection_results_{timestamp}.json')}")
    print(f"Blockchain logs saved to: {blockchain.log_file}")
    
    # Ask user if they want to keep the session running
    keep_running = input("\nKeep session running? (y/n): ").lower().strip() == 'y'
    
    if keep_running:
        print(f"Session {session.id} left running. You can delete it manually when finished.")
    else:
        # Stop session
        core.delete_session(session_id=session.id)
        print(f"Session {session.id} stopped and deleted.")
    
    print("MANET with ML simulation complete.")


if __name__ == "__main__":
    # Set paths
    pcap_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manet_pcaps")
    xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manet_session.xml")
    
    # Run the simulation
    create_manet_with_ml(pcap_dir=pcap_dir, xml_path=xml_path)

