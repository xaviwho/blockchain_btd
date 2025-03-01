from web3 import Web3
import json

# Blockchain connection
RPC_URL = "http://43.200.53.250:8548"
CONTRACT_ADDRESS = Web3.to_checksum_address("0x993e3cc1e8f252f51758f7d789f4039508c63a88")

# Contract ABI (Ensure this matches Remix ABI)
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

# Smart contract instance
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

def retrieve_logged_data():
    """Retrieve sensor logs from the blockchain."""
    try:
        total_entries = contract.functions.getLogCount().call()
        print(f"Total blockchain entries: {total_entries}")

        if total_entries == 0:
            print("⚠️ No data found on the blockchain. Ensure log_purechain.py was executed.")
            return

        sensor_data = []
        for i in range(total_entries):
            try:
                # Get the key from logKeys[]
                key = contract.functions.logKeys(i).call()
                
                # Now retrieve the log using the key
                entry = contract.functions.getLog(key).call()
                
                log = {
                    "sensor_id": entry[0],
                    "agent": entry[1],
                    "concentration": entry[2] / 1000,  # Convert back from int
                    "reaction_rate": entry[3] / 1000,
                    "timestamp": entry[4],
                }
                sensor_data.append(log)
            except Exception as e:
                print(f"❌ Error retrieving log at index {i}: {e}")
                continue  # Skip bad entries

        # Save retrieved data
        with open("data/retrieved_sensor_data.json", "w") as f:
            json.dump(sensor_data, f, indent=2)

        print("✅ Blockchain data retrieved and saved successfully.")
    except Exception as e:
        print(f"❌ Error retrieving blockchain data: {e}")

if __name__ == "__main__":
    retrieve_logged_data()
