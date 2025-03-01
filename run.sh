#!/bin/bash
echo "Generating sensor data..."
python3 src/sensor_synthesis/synthesize.py
echo "Logging data to Purechain..."
python3 src/blockchain/log_purechain.py