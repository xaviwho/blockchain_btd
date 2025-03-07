# AI-Augmented Blockchain for Biochemical Threat Detection in Disrupted Tactical MANETs

This repository contains the implementation and experimental framework for an AI-augmented blockchain system designed to detect biochemical threats in disrupted tactical Mobile Ad-hoc Networks (MANETs).

## Research Overview

This project addresses a critical gap in tactical network security by integrating machine learning-based threat detection with blockchain data integrity in degraded network environments. Our system is designed to:

- Detect nerve agents (such as sarin) using acetylcholinesterase-based sensor data
- Maintain data integrity through distributed blockchain logging
- Function effectively despite 20% packet loss and network disruptions
- Provide rapid threat detection with high confidence

## System Architecture

The system consists of three integrated components:

1. **Tactical MANET Simulation**: A 10-node wireless network with 20% packet loss, simulating degraded battlefield conditions
2. **Machine Learning Component**: Random Forest classification for rapid nerve agent detection (27.5ms average inference time)
3. **Blockchain Security Layer**: Distributed ledger for ensuring sensor data integrity and tamper resistance

## Key Results

- Successful detection of 32 sarin threats across 100 sensor readings
- 100% detection accuracy with high confidence scores
- Average inference time of 27.5ms, suitable for tactical response timeframes
- Resilient performance despite 20% simulated packet loss

## Repository Structure

```
blockchain_btd/
├── data/                     # Sensor data and blockchain records
├── ml/                       # Machine learning components
│   ├── models/               # Trained ML models
│   ├── data_preparation.py   # Data preprocessing
│   ├── models_sklearn.py     # Model architecture definitions
│   ├── train_sklearn.py      # Model training pipeline
│   ├── evaluation.py         # Performance evaluation
│   └── visualization.py      # Results visualization
├── manet_pcaps/              # Network capture files
├── results/                  # Performance results and visualizations
├── blockchain_logs/          # Blockchain transaction records
├── manet_output/             # MANET simulation output
├── manet.py                  # MANET simulation script
├── run_manet_only.py         # Standalone MANET script
├── run_ml_detection.py       # Standalone ML detection script
└── run_complete_system.sh    # Full system integration script
```

## Getting Started

### Prerequisites

- CORE Network Emulator
- Python 3.12+
- Required Python packages: scikit-learn, numpy, pandas, matplotlib, networkx, seaborn

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/blockchain-btd.git
cd blockchain-btd

# Create and activate virtual environment
python -m venv ml_venv
source ml_venv/bin/activate

# Install dependencies
pip install scikit-learn numpy pandas matplotlib seaborn networkx
```

### Running the System

```bash
# Run the complete system simulation
./run_complete_system.sh

# Visualize results
cd ml
python visualization.py
```

## Citation

If you use this code in your research, please cite:

```
@inproceedings{yourlastname2025ai,
  title={AI-Augmented Blockchain for Biochemical Threat Detection in Disrupted Tactical MANETs},
  author={Your Name and Coauthors},
  booktitle={MILCOM 2025 - IEEE Military Communications Conference},
  year={2025},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This research was conducted for MILCOM 2025
- Thanks to [your institution/lab] for support
