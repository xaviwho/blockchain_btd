from mininet.net import Mininet
from mininet.node import Controller, OVSKernelAP
from mininet.cli import CLI
from mininet.log import setLogLevel
from mininet.link import TCLink
import time

def simulate_manet():
    """Create a MANET (Mobile Ad-Hoc Network) with dynamic topology changes."""
    setLogLevel('info')
    
    print("🚀 Initializing MANET Simulation...")
    
    # Create Mininet-WiFi network
    net = Mininet(controller=Controller, link=TCLink, accessPoint=OVSKernelAP)
    
    # Add a controller
    net.addController('c0')
    
    # Add mobile nodes (simulating biochemical sensors)
    sta1 = net.addStation('sta1', ip='10.0.0.1', mac='00:00:00:00:01:01')
    sta2 = net.addStation('sta2', ip='10.0.0.2', mac='00:00:00:00:01:02')
    sta3 = net.addStation('sta3', ip='10.0.0.3', mac='00:00:00:00:01:03')
    
    # Add an access point (AP) for connectivity
    ap1 = net.addAccessPoint('ap1', ssid='manet', mode='g', channel='5')
    
    # Add a switch and a blockchain server (simulated as a host)
    s1 = net.addSwitch('s1')
    blockchain_server = net.addHost('h1', ip='10.0.0.10')
    
    # Establish links (wireless and wired)
    net.addLink(ap1, s1)
    net.addLink(blockchain_server, s1)
    net.addLink(sta1, ap1)
    net.addLink(sta2, ap1)
    net.addLink(sta3, ap1)
    
    # Start the network
    net.start()
    print("✅ MANET Simulation Running...")
    
    # Simulate dynamic network disruptions
    time.sleep(5)
    print("⚠️ Simulating packet loss and latency...")
    sta2.cmd('tc qdisc add dev sta2-wlan0 root netem loss 30% delay 100ms')  # Add 30% packet loss + 100ms delay
    time.sleep(5)
    
    print("🔄 Restoring network connectivity...")
    sta2.cmd('tc qdisc del dev sta2-wlan0 root netem')
    time.sleep(5)
    
    print("✅ MANET Simulation Complete.")
    CLI(net)  # Open CLI for manual control
    net.stop()

if __name__ == '__main__':
    simulate_manet()
