from web3 import Web3

RPC_URL = "http://43.200.53.250:8548"
CONTRACT_ADDRESS = Web3.to_checksum_address("0x993e3cc1e8f252f51758f7d789f4039508c63a88")

ABI = [
    {
        "inputs": [],
        "name": "getLogCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]

w3 = Web3(Web3.HTTPProvider(RPC_URL))

if not w3.is_connected():
    print(f"❌ Failed to connect to Purechain RPC at {RPC_URL}")
else:
    print("✅ Connected to Purechain!")

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

total_entries = contract.functions.getLogCount().call()
print(f"🔹 Total blockchain entries: {total_entries}")
