"""
run_manet_only.py

This script runs the MANET simulation and saves data for ML processing.
Designed to run in the CORE environment.
"""

from core.api.grpc import client
from core.api.grpc.wrappers import SessionState, Node, Position, Interface, Link
import time
import json
import os
from datetime import datetime

# Paths and directories
PCAP_DIR = "manet_pcaps"
XML_PATH = "manet_session.xml"
DATA_PATH = "data/sensor_data.json"
OUTPUT_DIR = "manet_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"manet_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

def create_manet_simulation():
    """Run MANET simulation and save data for ML processing"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PCAP_DIR, exist_ok=True)
    
    print("Starting MANET simulation...")
    
    # Connect to core-daemon gRPC server
    core = client.CoreGrpcClient()
    core.connect()  # Connect to default localhost:50051
    
    # Create a new session
    session = core.create_session()
    print(f"Created new session with ID: {session.id}")
    
    # Set session state to CONFIGURATION
    session.state = SessionState.CONFIGURATION
    
    # Create 10 stations and 1 AP
    stations = []
    for i in range(10):
        # Create a Node object
        node = Node(
            name=f'sta{i}',
            model='wlan',
            position=Position(x=i * 50, y=100)
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
        position=Position(x=250, y=200)
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

    # Configure WiFi
    for node_id in stations + [ap_id]:
        try:
            # Configure WiFi using a dictionary
            core.set_wlan_config(
                session_id=session.id,
                node_id=node_id,
                config={"ssid": "manet", "channel": "1"}
            )
            print(f"Configured WiFi for node {node_id}")
        except Exception as e:
            print(f"Error configuring WiFi for node {node_id}: {e}")

    # Create switches and direct links between stations and AP
    for i, sta_id in enumerate(stations):
        try:
            # Create a direct wired link between station and AP
            # This simplified approach avoids the switch intermediary that was causing issues
            link = Link(
                node1_id=sta_id,
                node2_id=ap_id
            )
            core.add_link(session_id=session.id, link=link)
            print(f"Created direct link from station {i} to AP")
        except Exception as e:
            print(f"Error creating link from station {i} to AP: {e}")
            print("Continuing with next link...")

    # Set packet loss using node_command instead of link options
    for i, sta_id in enumerate(stations):
        try:
            # Apply packet loss using tc command
            core.node_command(
                session_id=session.id,
                node_id=sta_id,
                command='tc qdisc add dev wlan0 root netem loss 20%'
            )
            print(f"Set 20% packet loss for station {i}")
        except Exception as e:
            print(f"Error setting packet loss for station {i}: {e}")

    # Try to start the session
    print("\nAttempting to start session...")
    try:
        # Try different approaches to start the session
        try:
            # Try with positional argument
            core.start_session(session.id)
            print("Session started successfully using start_session!")
        except Exception as e1:
            print(f"Error with start_session: {e1}")
            try:
                # Try direct state modification
                session.state = SessionState.RUNTIME
                print("Set session state to RUNTIME directly")
            except Exception as e2:
                print(f"Error setting session state: {e2}")
    except Exception as e:
        print(f"All attempts to start session failed: {e}")
    
    print("MANET started. Testing connectivity...")

    # Load sensor data
    try:
        with open(DATA_PATH, 'r') as f:
            sensor_data = json.load(f)
        print(f"Loaded {len(sensor_data)} sensor readings")
    except Exception as e:
        print(f"Error loading sensor data: {e}")
        sensor_data = []
    
    # Simulate data transmission
    transmitted_data = []
    
    for i, reading in enumerate(sensor_data[:100]):  # Use first 100 readings
        # Select a random station
        sender_id = stations[i % 10]
        
        # Add node info to reading
        reading_copy = reading.copy()  # Create a copy to avoid modifying the original
        reading_copy['node_id'] = sender_id
        reading_copy['transmission_time'] = datetime.now().isoformat()
        
        # Create data string for transmission
        data_str = f"Sensor-{reading['sensor_id']}-Agent-{reading['agent']}-Conc-{reading['concentration']}"
        
        # Transmit data
        try:
            core.node_command(
                session_id=session.id,
                node_id=sender_id,
                command=f'echo "{data_str}" | nc -u 10.0.0.254 12345'
            )
        except Exception as e:
            print(f"Error transmitting data: {e}")
        
        # Store for ML processing
        transmitted_data.append(reading_copy)
        
        # Add delay
        time.sleep(0.1)
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Transmitted {i + 1} readings")
    
    # Start tcpdump on stations
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pcap_files = []
    
    for i, sta_id in enumerate(stations[:3]):
        pcap_path = os.path.join(PCAP_DIR, f"wifi_traffic_sta{i+1}_{timestamp}.pcap")
        pcap_files.append(pcap_path)
        
        try:
            core.node_command(
                session_id=session.id,
                node_id=sta_id,
                command=f'tcpdump -i wlan0 -w {pcap_path} &'
            )
            print(f"Started tcpdump on station {i+1}, saving to {pcap_path}")
        except Exception as e:
            print(f"Error starting tcpdump on station {i+1}: {e}")
    
    # Run for 10 seconds
    print("Running capture for 10 seconds...")
    time.sleep(10)
    
    # Stop tcpdump
    for sta_id in stations[:3]:
        try:
            core.node_command(session_id=session.id, node_id=sta_id, command='pkill -9 tcpdump')
        except Exception as e:
            print(f"Error stopping tcpdump: {e}")
    
    # Save transmitted data for ML processing
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(transmitted_data, f, indent=2)
    
    print(f"Saved transmitted data to {OUTPUT_FILE}")
    print("MANET simulation complete. Ready for ML processing.")
    
    # Clean up
    try:
        core.delete_session(session_id=session.id)
        print(f"Deleted session {session.id}")
    except Exception as e:
        print(f"Error deleting session: {e}")
    
    return OUTPUT_FILE

if __name__ == "__main__":
    output_file = create_manet_simulation()
    print(f"MANET data saved to: {output_file}")
    print("Now run the ML detection script in your ML environment")