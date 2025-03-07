from core.api.grpc import client
from core.api.grpc.wrappers import SessionState, Node, Position, Interface, Link
import time
import json
import os
from datetime import datetime

def create_manet(pcap_dir=None, xml_path=None):
    """
    Create a MANET simulation with custom paths for output files.
    
    Args:
        pcap_dir: Directory to save pcap files (defaults to /tmp if None)
        xml_path: Path to save the session XML (defaults to /tmp/manet_session.xml if None)
    """
    # Set default paths if not provided
    if pcap_dir is None:
        pcap_dir = "/tmp"
    else:
        # Create directory if it doesn't exist
        os.makedirs(pcap_dir, exist_ok=True)
        
    if xml_path is None:
        xml_path = "/tmp/manet_session.xml"
    
    # Ensure parent directory for xml_path exists
    os.makedirs(os.path.dirname(os.path.abspath(xml_path)), exist_ok=True)
    
    print(f"PCAP files will be saved to: {pcap_dir}")
    print(f"Session XML will be saved to: {xml_path}")
    
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

    # Simulate PureChain data transmission
    try:
        with open("./blockchain_btd/data/sensor_data.json", "r") as f:
            sensor_data = json.load(f)
        
        print(f"Loaded sensor data with {len(sensor_data)} entries")
        
        for i, entry in enumerate(sensor_data[:50]):  # Use fewer entries for quicker testing
            sender_id = stations[i % 10]
            data = f"Data-{entry['sensor_id']}: {entry['concentration']}"
            core.node_command(session_id=session.id, node_id=sender_id, command=f'echo "{data}" | nc -u 10.0.0.254 12345')
            time.sleep(0.1)  # Reduced delay for testing
            
            # Print progress every 10 transmissions
            if (i + 1) % 10 == 0:
                print(f"Transmitted {i + 1} data packets")
    
    except Exception as e:
        print(f"Error simulating PureChain data: {e}")

    # Start tcpdump on each station (test with first 3 for now)
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

    print("\nSimulated PureChain data transmission. Capturing traffic with tcpdump...")
    
    # Run for 30 seconds
    print("Running simulation for 30 seconds...")
    time.sleep(30)

    # Stop tcpdump
    for sta_id in stations[:3]:
        core.node_command(session_id=session.id, node_id=sta_id, command='pkill -9 tcpdump')
    print("Stopped tcpdump on all stations")

    # Print summary
    print("\nMANET Simulation Summary:")
    print("=======================")
    print(f"Session ID: {session.id}")
    print(f"Total Nodes: {len(stations) + len(switches) + 1}")
    print("Captured traffic files:")
    for pcap_file in pcap_files:
        print(f" - {pcap_file}")
    print(f"Session XML saved to: {xml_path}")
    
    # Ask user if they want to keep the session running
    keep_running = input("\nKeep session running? (y/n): ").lower().strip() == 'y'
    
    if keep_running:
        print(f"Session {session.id} left running. You can delete it manually when finished.")
    else:
        # Stop session
        core.delete_session(session_id=session.id)
        print(f"Session {session.id} stopped and deleted.")
    
    print("MANET simulation complete.")

if __name__ == "__main__":
    # You can customize these paths
    custom_pcap_dir = os.path.expanduser("~/blockchain_btd/manet_pcaps")
    custom_xml_path = os.path.expanduser("~/blockchain_btd/manet_session.xml")
    
    # Use default paths (/tmp) if you don't want to customize
    # create_manet()
    
    # Or use custom paths
    create_manet(pcap_dir=custom_pcap_dir, xml_path=custom_xml_path)