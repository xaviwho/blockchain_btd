# AI-Enhanced Blockchain for Biochemical Threat Detection in Tactical MANETs
MILCOM 2025 project. Simulates biochemical sensor data, secures it via blockchain, and predicts threats with AI over a tactical MANET.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Install Docker & Docker Compose: `sudo apt install docker.io docker-compose`
3. Run sensor synthesis: `python src/sensor_synthesis/synthesize.py`
4. Start Fabric network & log data: `bash run.sh`

## Notes
- Ensure Docker runs without sudo: `sudo usermod -aG docker $USER` (relogin after).