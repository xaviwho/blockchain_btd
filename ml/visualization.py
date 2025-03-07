"""
visualization.py - Updated for split MANET/ML architecture

This module generates advanced visualizations for nerve agent detection results.
It creates publication-quality figures for research papers and presentations.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter1d
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
import re

# Constants
RESULTS_DIR = "/home/vboxuser/blockchain_btd/results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PAPER_DIR = os.path.join(RESULTS_DIR, "paper_figures")
DATA_PATH = "/home/vboxuser/blockchain_btd/data/sensor_data.json"
MANET_OUTPUT_DIR = "/home/vboxuser/blockchain_btd/manet_output"
BLOCKCHAIN_LOGS_DIR = "/home/vboxuser/blockchain_btd/blockchain_logs"
MODEL_TYPE = "rf"


def setup_directories():
    """Create necessary directories for output figures"""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(PAPER_DIR, exist_ok=True)


def set_publication_style():
    """Set matplotlib style for publication-quality figures"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 300
    # Use system default fonts instead of Times New Roman
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.alpha'] = 0.7


def load_data():
    """Load all available data for visualization"""
    data = {}
    
    # Load raw sensor data
    try:
        with open(DATA_PATH, 'r') as f:
            data['sensor_data'] = json.load(f)
        print(f"Loaded {len(data['sensor_data'])} raw sensor readings from {DATA_PATH}")
    except Exception as e:
        print(f"Error loading sensor data: {e}")
        data['sensor_data'] = []
    
    # Load MANET output data (most recent file)
    try:
        manet_files = glob.glob(os.path.join(MANET_OUTPUT_DIR, "manet_data_*.json"))
        if manet_files:
            latest_manet_file = max(manet_files, key=os.path.getmtime)
            with open(latest_manet_file, 'r') as f:
                data['manet_data'] = json.load(f)
            print(f"Loaded {len(data['manet_data'])} MANET transmitted readings from {latest_manet_file}")
        else:
            print("No MANET data files found")
            data['manet_data'] = []
    except Exception as e:
        print(f"Error loading MANET data: {e}")
        data['manet_data'] = []
    
    # Load detection results (most recent file)
    try:
        detection_files = glob.glob(os.path.join(RESULTS_DIR, "detection_results_*.json"))
        if detection_files:
            latest_detection_file = max(detection_files, key=os.path.getmtime)
            with open(latest_detection_file, 'r') as f:
                data['detection_results'] = json.load(f)
            print(f"Loaded {len(data['detection_results'])} detection results from {latest_detection_file}")
        else:
            print("No detection results found")
            data['detection_results'] = []
    except Exception as e:
        print(f"Error loading detection results: {e}")
        data['detection_results'] = []
    
    # Load blockchain logs (most recent file)
    try:
        blockchain_files = glob.glob(os.path.join(BLOCKCHAIN_LOGS_DIR, "blockchain_log_*.json"))
        if blockchain_files:
            latest_blockchain_file = max(blockchain_files, key=os.path.getmtime)
            with open(latest_blockchain_file, 'r') as f:
                data['blockchain_logs'] = json.load(f)
            print(f"Loaded {len(data['blockchain_logs'])} blockchain logs from {latest_blockchain_file}")
        else:
            print("No blockchain logs found")
            data['blockchain_logs'] = []
    except Exception as e:
        print(f"Error loading blockchain logs: {e}")
        data['blockchain_logs'] = []
    
    return data


def plot_sensor_data_overview(data, save_dir=FIGURES_DIR):
    """
    Create an overview plot of the sensor data
    
    Args:
        data: Dictionary containing data
        save_dir: Directory to save the figure
    """
    if not data.get('sensor_data'):
        print("No sensor data to plot")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(data['sensor_data'])
    
    # Convert timestamp to datetime if it's not already
    if isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Plot concentration distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['concentration'], kde=True, bins=30)
    plt.title('Distribution of Nerve Agent Concentrations')
    plt.xlabel('Concentration')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'concentration_distribution.png'))
    plt.close()
    
    # Plot reaction rate vs concentration
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='concentration', y='reaction_rate', hue='agent', s=70, alpha=0.7)
    plt.title('Reaction Rate vs Concentration by Agent Type')
    plt.xlabel('Concentration')
    plt.ylabel('Reaction Rate')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reaction_rate_vs_concentration.png'))
    plt.close()
    
    # If we have detection results, overlay detection threshold
    if data.get('detection_results'):
        detected = [r for r in data['detection_results'] if r.get('threat_detected', False)]
        if detected:
            # Extract threshold based on detected samples
            min_detected = min([r.get('concentration', 0) for r in detected])
            
            plt.figure(figsize=(10, 6))
            sns.histplot(df['concentration'], kde=True, bins=30)
            plt.axvline(x=min_detected, color='red', linestyle='--', 
                       label=f'Detection Threshold (~{min_detected:.4f})')
            plt.title('Nerve Agent Concentration Distribution with Detection Threshold')
            plt.xlabel('Concentration')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'concentration_with_threshold.png'))
            plt.close()


def plot_detection_results(data, save_dir=FIGURES_DIR):
    """
    Plot nerve agent detection results
    
    Args:
        data: Dictionary containing detection results
        save_dir: Directory to save the figure
    """
    if not data.get('detection_results'):
        print("No detection results to plot")
        return
    
    # Create DataFrame from detection results
    df = pd.DataFrame(data['detection_results'])
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by sensor_id for visualization
    df = df.sort_values('sensor_id')
    
    # 1. Detection Probability Heatmap
    plt.figure(figsize=(12, 8))
    
    # Create a matrix for the heatmap (rows=sensor_id, cols=sequential order)
    sensor_ids = sorted(df['sensor_id'].unique())
    
    # Initialize the data matrix
    data_matrix = np.zeros((len(sensor_ids), len(df)))
    
    # Fill the matrix with threat probabilities
    for i, sensor_id in enumerate(sensor_ids):
        sensor_df = df[df['sensor_id'] == sensor_id]
        if not sensor_df.empty:
            for j, (_, row) in enumerate(sensor_df.iterrows()):
                col_idx = j % data_matrix.shape[1]  # In case we have more readings than columns
                data_matrix[i, col_idx] = row.get('threat_probability', 0)
    
    # Plot the heatmap - save the mappable for colorbar
    heatmap = sns.heatmap(data_matrix, cmap='YlOrRd', vmin=0, vmax=1)
    plt.title('Threat Detection Probability by Sensor')
    plt.xlabel('Sample Sequence')
    plt.ylabel('Sensor ID')
    plt.colorbar(mappable=heatmap.get_children()[0], label='Threat Probability')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detection_heatmap.png'))
    plt.close()
    
    # 2. Detection summary by agent type
    if 'agent' in df.columns:
        agent_counts = df.groupby(['agent', 'threat_detected']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(10, 6))
        agent_counts.plot(kind='bar', stacked=True)
        plt.title('Detection Results by Agent Type')
        plt.xlabel('Agent')
        plt.ylabel('Count')
        plt.legend(['Not Detected', 'Detected'])
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'detection_by_agent.png'))
        plt.close()
    
    # 3. Inference time distribution
    if 'inference_time_ms' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['inference_time_ms'], kde=True, bins=20)
        plt.axvline(x=df['inference_time_ms'].mean(), color='red', linestyle='--',
                   label=f'Mean: {df["inference_time_ms"].mean():.2f} ms')
        plt.title('Inference Time Distribution')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'inference_time_distribution.png'))
        plt.close()
    
    # 4. ROC curve (simulated since we don't have ground truth)
    # For demonstration, we'll use concentration as proxy for ground truth
    if 'concentration' in df.columns and 'threat_probability' in df.columns:
        # Sort by concentration
        df_sorted = df.sort_values('concentration')
        
        # Calculate TPR and FPR at different thresholds
        thresholds = np.linspace(0, 1, 100)
        tpr = []
        fpr = []
        
        # Define ground truth based on concentration threshold (top 30% are positive)
        concentration_threshold = df_sorted['concentration'].quantile(0.7)
        df_sorted['true_positive'] = df_sorted['concentration'] > concentration_threshold
        
        for threshold in thresholds:
            df_sorted['predicted_positive'] = df_sorted['threat_probability'] > threshold
            
            # True positives: predicted positive and actually positive
            tp = sum((df_sorted['predicted_positive']) & (df_sorted['true_positive']))
            # False positives: predicted positive but actually negative
            fp = sum((df_sorted['predicted_positive']) & (~df_sorted['true_positive']))
            # True negatives: predicted negative and actually negative
            tn = sum((~df_sorted['predicted_positive']) & (~df_sorted['true_positive']))
            # False negatives: predicted negative but actually positive
            fn = sum((~df_sorted['predicted_positive']) & (df_sorted['true_positive']))
            
            # Calculate rates
            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, 'b-', linewidth=2)
        plt.plot([0, 1], [0, 1], 'r--')  # Random classifier line
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        
        # Calculate AUC
        auc = np.trapz(tpr, fpr)
        plt.annotate(f'AUC = {auc:.3f}', xy=(0.6, 0.2), xycoords='data',
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'roc_curve.png'))
        plt.close()


def plot_tactical_manet_diagram(save_dir=PAPER_DIR):
    """
    Create a diagram of the tactical MANET with simulated biochemical threat
    
    Args:
        save_dir: Directory to save the figure
    """
    # Set a simpler font at the beginning of the function
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create a new figure
    plt.figure(figsize=(12, 10))
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes: stations, AP, and switches
    stations = [f'Station {i+1}' for i in range(10)]
    
    # Add nodes with positions
    # Stations in a ring
    for i, station in enumerate(stations):
        angle = 2 * np.pi * i / len(stations)
        x = 6 * np.cos(angle)
        y = 6 * np.sin(angle)
        G.add_node(station, pos=(x, y), type='station')
    
    # AP in the center
    G.add_node('AP', pos=(0, 0), type='ap')
    
    # Add edges (direct connection to AP with 20% packet loss)
    for station in stations:
        G.add_edge(station, 'AP', type='wireless', loss=0.2)
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the network
    plt.gca().set_aspect('equal')
    
    # Draw nodes by type
    station_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'station']
    ap_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'ap']
    
    # Draw AP
    nx.draw_networkx_nodes(G, pos, nodelist=ap_nodes, node_size=1500, 
                           node_color='red', node_shape='s', alpha=0.9)
    
    # Draw stations
    nx.draw_networkx_nodes(G, pos, nodelist=station_nodes, node_size=1000, 
                           node_color='lightblue', node_shape='o', alpha=0.8)
    
    # Draw edges (all wireless with 20% packet loss)
    wireless_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'wireless']
    
    nx.draw_networkx_edges(G, pos, edgelist=wireless_edges, width=1.5, 
                           alpha=0.7, style='dashed', edge_color='blue')
    
    # Add labels - specify font_family as sans-serif here
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Add a simulated biochemical threat area
    threat_circle = plt.Circle((4, 4), 3, color='yellow', alpha=0.2, label='Threat Area')
    plt.gca().add_patch(threat_circle)
    
    # Add text for packet loss
    plt.text(0, -8, "All wireless links: 20% packet loss", 
             ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add blockchain blocks
    for i in range(5):
        block = plt.Rectangle((7, i-5), 1.5, 0.8, facecolor='lightgreen', 
                             alpha=0.8, edgecolor='green', linewidth=1)
        plt.gca().add_patch(block)
        plt.text(7.75, i-4.6, f"Block {i+1}", ha='center', va='center', fontsize=8)
    
    # Add connecting lines between blocks
    for i in range(4):
        plt.plot([7.75, 7.75], [i-4.2, i-4], 'g-', linewidth=1)
    
    # Add ML models on stations
    for i, station in enumerate(stations[:3]):  # Add ML to first 3 stations
        station_pos = pos[station]
        ml_circle = plt.Circle((station_pos[0] + 0.8, station_pos[1] + 0.8), 0.4, 
                               facecolor='purple', alpha=0.7, edgecolor='black')
        plt.gca().add_patch(ml_circle)
        plt.text(station_pos[0] + 0.8, station_pos[1] + 0.8, "ML", 
                 ha='center', va='center', color='white', fontsize=8)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label='Sensor Node'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=15, label='Access Point'),
        plt.Line2D([0], [0], linestyle='dashed', color='blue', linewidth=1.5, label='Wireless Link (20% Loss)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='ML Model'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', markersize=15, label='Blockchain'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', alpha=0.2, markersize=15, label='Threat Area')
    ]
    
    plt.legend(handles=legend_elements, loc='lower right')
    
    # Set title and remove axes
    plt.title('Tactical MANET with Blockchain and ML-based Biochemical Threat Detection')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'tactical_manet_diagram.png'), dpi=300)
    plt.close()
    
    # Reset font settings at the end of the function
    plt.rcParams['font.family'] = plt.rcParams.get('font.family', 'sans-serif')


def plot_blockchain_analysis(data, save_dir=PAPER_DIR):
    """
    Analyze and visualize blockchain data
    
    Args:
        data: Dictionary containing blockchain logs
        save_dir: Directory to save the figure
    """
    if not data.get('blockchain_logs'):
        print("No blockchain logs to analyze")
        return
    
    # Extract blockchain data
    blockchain_data = data['blockchain_logs']
    
    # Convert to DataFrame
    df = pd.DataFrame(blockchain_data)
    
    # Convert timestamps if needed
    if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], str):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    # Identify readings vs detection results
    df['entry_type'] = df.apply(
        lambda row: 'Detection Result' if 'detection_result' in row else 'Sensor Reading', 
        axis=1
    )
    
    # Plot transaction types
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='entry_type')
    plt.title('Blockchain Transaction Types')
    plt.xlabel('Transaction Type')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'blockchain_transaction_types.png'))
    plt.close()
    
    # Plot transactions over time
    if 'timestamp' in df.columns:
        plt.figure(figsize=(12, 6))
        df['minute'] = df['timestamp'].dt.floor('min')
        transaction_counts = df.groupby(['minute', 'entry_type']).size().unstack(fill_value=0)
        transaction_counts.plot(kind='line', marker='o')
        plt.title('Blockchain Transactions Over Time')
        plt.xlabel('Time')
        plt.ylabel('Number of Transactions')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'blockchain_transactions_timeline.png'))
        plt.close()
    
    # Plot blockchain growth
    plt.figure(figsize=(10, 6))
    df['cumulative_blocks'] = range(1, len(df) + 1)
    df.plot(x='block_number', y='cumulative_blocks', figsize=(10, 6))
    plt.title('Blockchain Growth')
    plt.xlabel('Block Number')
    plt.ylabel('Total Blocks')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'blockchain_growth.png'))
    plt.close()
    
    # Distribution of transactions by node
    if 'node_id' in df.columns:
        plt.figure(figsize=(12, 6))
        node_counts = df.groupby(['node_id', 'entry_type']).size().unstack(fill_value=0)
        node_counts.plot(kind='bar', stacked=True)
        plt.title('Blockchain Transactions by Node')
        plt.xlabel('Node ID')
        plt.ylabel('Number of Transactions')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'blockchain_transactions_by_node.png'))
        plt.close()


def plot_system_architecture(save_dir=PAPER_DIR):
    """
    Create a diagram of the system architecture
    
    Args:
        save_dir: Directory to save the figure
    """
    # Create figure
    fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
    
    # Define component positions
    components = {
        'Sensors': (1, 7),
        'Data Preprocessing': (3, 7),
        f'{MODEL_TYPE.upper()} Model': (5, 7),
        'Threat Detection': (7, 7),
        'Alert System': (9, 7),
        
        'Blockchain Validation': (5, 5),
        'Data Integrity Check': (3, 5),
        'Distributed Ledger': (7, 5),
        
        'MANET Simulation': (1, 3),
        'Packet Routing (OLSR)': (3, 3),
        'Network Disruption': (5, 3),
        'Resilience Testing': (7, 3),
        'Performance Metrics': (9, 3),
        
        'Station 1': (2, 1),
        'Station 2': (4, 1),
        'Station 3': (6, 1),
        'Access Point': (8, 1),
    }
    
    # Define connections
    connections = [
        ('Sensors', 'Data Preprocessing'),
        ('Data Preprocessing', f'{MODEL_TYPE.upper()} Model'),
        (f'{MODEL_TYPE.upper()} Model', 'Threat Detection'),
        ('Threat Detection', 'Alert System'),
        
        ('Data Preprocessing', 'Data Integrity Check'),
        ('Data Integrity Check', 'Blockchain Validation'),
        ('Blockchain Validation', 'Distributed Ledger'),
        ('Distributed Ledger', 'Threat Detection'),
        
        ('MANET Simulation', 'Packet Routing (OLSR)'),
        ('Packet Routing (OLSR)', 'Network Disruption'),
        ('Network Disruption', 'Resilience Testing'),
        ('Resilience Testing', 'Performance Metrics'),
        
        ('MANET Simulation', 'Station 1'),
        ('MANET Simulation', 'Station 2'),
        ('MANET Simulation', 'Station 3'),
        ('MANET Simulation', 'Access Point'),
        
        ('Station 1', 'Packet Routing (OLSR)'),
        ('Station 2', 'Packet Routing (OLSR)'),
        ('Station 3', 'Packet Routing (OLSR)'),
        ('Access Point', 'Packet Routing (OLSR)'),
    ]
    
    # Define layers
    layers = {
        'Biochemical Detection Layer': ['Sensors', 'Data Preprocessing', f'{MODEL_TYPE.upper()} Model', 'Threat Detection', 'Alert System'],
        'Blockchain Security Layer': ['Data Integrity Check', 'Blockchain Validation', 'Distributed Ledger'],
        'Network Simulation Layer': ['MANET Simulation', 'Packet Routing (OLSR)', 'Network Disruption', 'Resilience Testing', 'Performance Metrics'],
        'Node Layer': ['Station 1', 'Station 2', 'Station 3', 'Access Point']
    }
    
    # Define layer colors
    layer_colors = {
        'Biochemical Detection Layer': 'lightblue',
        'Blockchain Security Layer': 'lightgreen',
        'Network Simulation Layer': 'lightyellow',
        'Node Layer': 'lightgray'
    }
    
    # Draw layer backgrounds
    for layer_name, components_list in layers.items():
        # Get coordinates of components in this layer
        coords = [components[c] for c in components_list]
        x_min = min([x for x, y in coords]) - 0.7
        x_max = max([x for x, y in coords]) + 0.7
        y_min = min([y for x, y in coords]) - 0.4
        y_max = max([y for x, y in coords]) + 0.4
        
        # Draw rectangle
        rect = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                            facecolor=layer_colors[layer_name], alpha=0.3,
                            edgecolor='gray', linewidth=1, zorder=0)
        ax.add_patch(rect)
        
        # Add layer name
        ax.text(x_min + 0.1, y_max - 0.2, layer_name, fontsize=12, 
                fontweight='bold', ha='left', va='top')
    
    # Draw connections
    for start, end in connections:
        start_pos = components[start]
        end_pos = components[end]
        
        # Draw arrow
        ax.annotate("", xy=end_pos, xytext=start_pos,
                   arrowprops=dict(arrowstyle="->", lw=1.5, color='gray'))
    
    # Draw components
    for name, pos in components.items():
        # Determine which layer this component belongs to
        for layer_name, components_list in layers.items():
            if name in components_list:
                color = layer_colors[layer_name]
                break
        else:
            color = 'white'
        
        # Draw node
        circle = plt.Circle(pos, 0.5, facecolor=color, edgecolor='black', 
                           linewidth=1, alpha=0.8, zorder=10)
        ax.add_patch(circle)
        
        # Add text
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=9,
               fontweight='bold', zorder=20)
    
    # Set limits and remove axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.set_title('AI-Augmented Blockchain for Biochemical Threat Detection System Architecture',
                fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'system_architecture.png'), dpi=300)
    plt.close()


def plot_summary_metrics(data, save_dir=PAPER_DIR):
    """
    Create a summary visualization of key performance metrics
    
    Args:
        data: Dictionary containing all results
        save_dir: Directory to save the figure
    """
    # Initialize metrics
    metrics = {}
    
    # Extract detection statistics if available
    if data.get('detection_results'):
        detection_results = data['detection_results']
        
        # Total readings
        metrics['Total Readings'] = len(detection_results)
        
        # Threats detected
        threat_count = sum(1 for r in detection_results if r.get('threat_detected', False))
        metrics['Threats Detected'] = threat_count
        metrics['Detection Rate'] = f"{threat_count/len(detection_results)*100:.1f}%"
        
        # Average inference time
        if all('inference_time_ms' in r for r in detection_results):
            avg_time = np.mean([r['inference_time_ms'] for r in detection_results])
            metrics['Avg. Inference Time'] = f"{avg_time:.2f} ms"
        
        # Confidence statistics
        if all('threat_probability' in r for r in detection_results):
            probs = [r['threat_probability'] for r in detection_results if r.get('threat_detected', False)]
            if probs:
                metrics['Avg. Detection Confidence'] = f"{np.mean(probs)*100:.1f}%"
    
    # Network statistics
    metrics['Packet Loss Rate'] = "20.0%"
    metrics['Network Topology'] = "10 stations, 1 AP"
    
    # Create a table

def plot_summary_metrics(data, save_dir=PAPER_DIR):
    """
    Create a summary visualization of key performance metrics
    
    Args:
        data: Dictionary containing all results
        save_dir: Directory to save the figure
    """
    # Initialize metrics
    metrics = {}
    
    # Extract detection statistics if available
    if data.get('detection_results'):
        detection_results = data['detection_results']
        
        # Total readings
        metrics['Total Readings'] = len(detection_results)
        
        # Threats detected
        threat_count = sum(1 for r in detection_results if r.get('threat_detected', False))
        metrics['Threats Detected'] = threat_count
        metrics['Detection Rate'] = f"{threat_count/len(detection_results)*100:.1f}%"
        
        # Average inference time
        if all('inference_time_ms' in r for r in detection_results):
            avg_time = np.mean([r['inference_time_ms'] for r in detection_results])
            metrics['Avg. Inference Time'] = f"{avg_time:.2f} ms"
        
        # Confidence statistics
        if all('threat_probability' in r for r in detection_results):
            probs = [r['threat_probability'] for r in detection_results if r.get('threat_detected', False)]
            if probs:
                metrics['Avg. Detection Confidence'] = f"{np.mean(probs)*100:.1f}%"
    
    # Network statistics
    metrics['Packet Loss Rate'] = "20.0%"
    metrics['Network Topology'] = "10 stations, 1 AP"
    
    # Create a table visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = [[k, v] for k, v in metrics.items()]
    
    # Create table
    table = ax.table(cellText=table_data, 
                    colLabels=['Metric', 'Value'],
                    loc='center',
                    cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Set properties for header cells
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Set alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#D9E1F2')
    
    plt.title('System Performance Metrics Summary', fontsize=16, pad=20)
    plt.savefig(os.path.join(save_dir, 'performance_metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main visualization pipeline"""
    # Set up directories
    setup_directories()
    
    # Set matplotlib style for publication
    set_publication_style()
    
    print("=== Generating Visualizations for Research Paper ===")
    
    # Load data
    data = load_data()
    
    # Generate visualizations with error handling
    try:
        print("\nGenerating sensor data overview...")
        plot_sensor_data_overview(data)
    except Exception as e:
        print(f"Error generating sensor data overview: {e}")
    
    try:
        print("Generating detection results visualizations...")
        plot_detection_results(data)
    except Exception as e:
        print(f"Error generating detection results: {e}")
    
    try:
        print("Generating blockchain analysis...")
        plot_blockchain_analysis(data)
    except Exception as e:
        print(f"Error generating blockchain analysis: {e}")
    
    try:
        print("Generating tactical MANET diagram...")
        plot_tactical_manet_diagram()
    except Exception as e:
        print(f"Error generating tactical MANET diagram: {e}")
    
    try:
        print("Generating system architecture diagram...")
        plot_system_architecture()
    except Exception as e:
        print(f"Error generating system architecture diagram: {e}")
    
    try:
        print("Generating summary metrics visualization...")
        plot_summary_metrics(data)
    except Exception as e:
        print(f"Error generating summary metrics: {e}")
    
    print("\nVisualization complete! Figures saved to:")
    print(f"- General visualizations: {FIGURES_DIR}")
    print(f"- Paper figures: {PAPER_DIR}")


if __name__ == "__main__":
    main()

