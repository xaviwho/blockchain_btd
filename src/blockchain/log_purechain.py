import json
import time
from web3 import Web3

# Purechain config
RPC_URL = "http://43.200.53.250:8548"
ACCOUNT = "0x1D8898f42aC1330eE879e10Bf27Ca723A6B1649f"
PRIVATE_KEY = "a471850a08d06bcc47850274208275f1971c9f5888bd0a08fbc680ed9701cfda"
CONTRACT_ADDRESS = Web3.to_checksum_address("0x993e3cc1e8f252f51758f7d789f4039508c63a88")

# Contract ABI
ABI = [
	{
		"anonymous": False,
		"inputs": [
			{
				"indexed": False,
				"internalType": "string",
				"name": "key",
				"type": "string"
			},
			{
				"indexed": False,
				"internalType": "uint256",
				"name": "sensorId",
				"type": "uint256"
			},
			{
				"indexed": False,
				"internalType": "string",
				"name": "agent",
				"type": "string"
			}
		],
		"name": "DataLogged",
		"type": "event"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "_sensorId",
				"type": "uint256"
			},
			{
				"internalType": "string",
				"name": "_agent",
				"type": "string"
			},
			{
				"internalType": "uint256",
				"name": "_concentration",
				"type": "uint256"
			},
			{
				"internalType": "uint256",
				"name": "_reactionRate",
				"type": "uint256"
			},
			{
				"internalType": "string",
				"name": "_timestamp",
				"type": "string"
			}
		],
		"name": "logData",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "key",
				"type": "string"
			}
		],
		"name": "getLog",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			},
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			},
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			},
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			},
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "getLogCount",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"name": "logKeys",
		"outputs": [
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			}
		],
		"name": "logs",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "sensorId",
				"type": "uint256"
			},
			{
				"internalType": "string",
				"name": "agent",
				"type": "string"
			},
			{
				"internalType": "uint256",
				"name": "concentration",
				"type": "uint256"
			},
			{
				"internalType": "uint256",
				"name": "reactionRate",
				"type": "uint256"
			},
			{
				"internalType": "string",
				"name": "timestamp",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	}
]

w3 = Web3(Web3.HTTPProvider(RPC_URL))
if not w3.is_connected():
    raise Exception(f"Failed to connect to Purechain RPC at {RPC_URL}")

def log_sensor_data(data_file="./data/sensor_data.json"):
    """Log sensor data to Purechain."""
    contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)
    with open(data_file, "r") as f:
        sensor_data = json.load(f)
    
    for entry in sensor_data[:]:
        try:
            nonce = w3.eth.get_transaction_count(ACCOUNT, "pending")  # Use 'pending' to prevent duplicate nonce
            gas_price = int(w3.eth.gas_price * 1.2)  # Increase gas price dynamically
            
            tx = contract.functions.logData(
                entry["sensor_id"],
                entry["agent"],
                int(entry["concentration"] * 1000),
                int(entry["reaction_rate"] * 1000),
                entry["timestamp"]
            ).build_transaction({
                "from": ACCOUNT,
                "nonce": nonce,
                "gas": 200000,
                "gasPrice": gas_price
            })
            
            signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
            tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            print(f"Logged entry {entry['sensor_id']}-{entry['timestamp']} with tx hash: {tx_hash.hex()}")
            time.sleep(2)  # Add slight delay to prevent nonce conflicts
        
        except Exception as e:
            print(f"Error logging entry {entry['sensor_id']}: {e}")

if __name__ == "__main__":
    log_sensor_data()
