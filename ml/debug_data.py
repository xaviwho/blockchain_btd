"""
Debug script to understand what's happening with the data preparation
"""

import json
import pandas as pd
import numpy as np

# Load the data
data_path = "/home/vboxuser/blockchain_btd/data/sensor_data.json"
print(f"Loading data from {data_path}")

with open(data_path, 'r') as f:
    sensor_data = json.load(f)

print(f"Loaded {len(sensor_data)} sensor readings")

# Check the structure of the data
print("\nSample data entry:")
print(json.dumps(sensor_data[0], indent=2))

# Convert to DataFrame
df = pd.DataFrame(sensor_data)
print("\nDataFrame columns:", df.columns.tolist())
print("DataFrame shape:", df.shape)

# Check if required columns exist
required_columns = ['sensor_id', 'concentration', 'timestamp']
for col in required_columns:
    print(f"Column '{col}' exists: {col in df.columns}")

# Check unique sensor IDs
sensor_ids = df['sensor_id'].unique()
print(f"\nNumber of unique sensors: {len(sensor_ids)}")
print("Sensor IDs:", sensor_ids)

# Check number of readings per sensor
for sensor in sensor_ids:
    count = len(df[df['sensor_id'] == sensor])
    print(f"Sensor {sensor}: {count} readings")

print("\nData inspection complete")