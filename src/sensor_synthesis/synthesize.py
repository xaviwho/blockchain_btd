# src/sensor_synthesis/synthesize.py
import numpy as np
import json
from datetime import datetime, timedelta

# Michaelis-Menten parameters
V_MAX = 1.0          # Max reaction rate (arbitrary units)
K_M = 0.1            # Michaelis constant (ppm)
SUBSTRATE_CONC = 1.0 # Fixed substrate concentration (ppm)
K_I = 0.05           # Inhibition constant (ppm)
BASE_CONC = 0.05     # Initial sarin concentration (ppm)
ESCALATION_RATE = 0.002  # Sarin increase per cycle (ppm/s)
NOISE_SCALE = 0.02   # Gaussian noise scale (ppm)

def calculate_reaction_rate(substrate_conc, inhibitor_conc):
    """Compute reaction rate with competitive inhibition."""
    if inhibitor_conc > 0:
        k_m_app = K_M * (1 + inhibitor_conc / K_I)
        return V_MAX * substrate_conc / (k_m_app + substrate_conc)
    return V_MAX * substrate_conc / (K_M + substrate_conc)

def generate_sarin_concentration(t):
    """Simulate escalating sarin concentration with noise."""
    return BASE_CONC + ESCALATION_RATE * t + np.random.normal(0, NOISE_SCALE)

def synthesize_sensor_data(num_sensors=10, cycles=100):
    """Generate synthetic sensor data with unique sensor IDs."""
    start_time = datetime.now()
    sensor_data = []
    global_sensor_id = 1  # Start with sensor ID 1

    for sensor in range(num_sensors):
        for t in range(cycles):
            inhibitor_conc = generate_sarin_concentration(t)
            reaction_rate = calculate_reaction_rate(SUBSTRATE_CONC, inhibitor_conc)
            timestamp = (start_time + timedelta(seconds=t)).isoformat()
            entry = {
                "sensor_id": global_sensor_id,  # Unique ID per entry
                "agent": "sarin",
                "concentration": round(inhibitor_conc, 3),
                "reaction_rate": round(reaction_rate, 3),
                "timestamp": timestamp
            }
            sensor_data.append(entry)
            global_sensor_id += 1  # Increment ID for uniqueness
    
    return sensor_data

def save_data(data, filepath="./data/sensor_data.json"):
    """Save synthesized data to JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    # Generate and save data
    data = synthesize_sensor_data()
    save_data(data)
    print(f"Generated {len(data)} data points. Saved to data/sensor_data.json")
    print("Sample entry:", json.dumps(data[0], indent=2))
