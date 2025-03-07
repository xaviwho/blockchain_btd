#!/bin/bash

# Run MANET simulation in CORE environment
echo "Starting MANET simulation..."
source /opt/core/venv/bin/activate
python run_manet_only.py
deactivate

# Run ML detection in ML environment
echo "Starting ML detection..."
source ~/blockchain_btd/ml/ml_venv/bin/activate
python run_ml_detection.py
deactivate

echo "Complete system simulation finished!"