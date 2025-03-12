import torch
from models import Net
from server import FederatedServer
from clients import FederatedClient  # Keeping as clients as requested
from blockchain_connector import BlockchainConnector
from web3 import Web3 
import copy
import os

def test_blockchain_integration():
    # First, check if ABI files exist
    comm_abi_path = "blockchain/FederatedLearningCommunication.json"
    contrib_abi_path = "blockchain/FederatedLearningContribution.json"
    
    if not os.path.exists(comm_abi_path) or not os.path.exists(contrib_abi_path):
        print(f"Warning: ABI files not found at {comm_abi_path} or {contrib_abi_path}")
        print("Using absolute paths instead")
        comm_abi_path = "C:/Users/eleor/OneDrive/Bureau/MILA/Semestre 2/IFT6056-blockchain/Projet/FL_ver1/artifacts/contracts/FLCommunication.sol/FLCommunication.json"
        contrib_abi_path = "C:/Users/eleor/OneDrive/Bureau/MILA/Semestre 2/IFT6056-blockchain/Projet/FL_ver1/artifacts/contracts/FLContribution.sol/FLContribution.json"

    # Initialize Web3
    w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
    print(f"Connected to Ethereum node: {w3.is_connected()}")
    print(f"Current block number: {w3.eth.block_number}")
    
    # Initialize a simple model - keeping your parameters as requested
    model = Net(input_dim=1, hidden_dim=64, output_dim=10)
    
    # Initialize blockchain connector with checksummed addresses
    blockchain = BlockchainConnector(
        provider_url="http://localhost:8545",
        comm_contract_address=w3.to_checksum_address("0xe7f1725e7734ce288f8367e1bb143e90bb3f0512"),
        contrib_contract_address=w3.to_checksum_address("0x9fe46736679d2d9a65f0992f2272de9f3c7fa6e0"),
        comm_abi_path=comm_abi_path,
        contrib_abi_path=contrib_abi_path
    )
    
    # The rest of your code remains the same
    # Initialize server with blockchain
    server = FederatedServer(
        global_model=model,
        blockchain_enabled=True
    )
    server.blockchain = blockchain
    
    # Initialize two test clients
    client1 = FederatedClient(client_id=1, model=copy.deepcopy(model), data_loader=None)
    client2 = FederatedClient(client_id=2, model=copy.deepcopy(model), data_loader=None)
    
    # Get client addresses
    client_addresses = [
        w3.to_checksum_address("0x70997970C51812dc3A010C7d01b50e0d17dc79C8"),  # Account 1
        w3.to_checksum_address("0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"),  # Account 2
    ]
    
    # Test round initiation
    print("Testing round initiation...")
    round_id = server.start_round("test_public_key")
    print(f"Started round {round_id}")
    
    # Test contribution recording
    print("\nTesting contribution recording...")
    server.blockchain.record_contribution(
        participant=client_addresses[0],
        round_id=round_id,
        contribution_score=75.5
    )
    server.blockchain.record_contribution(
        participant=client_addresses[1],
        round_id=round_id,
        contribution_score=82.3
    )
    
    # Test retrieving contributions
    print("\nTesting contribution retrieval...")
    participants, contributions = server.blockchain.get_round_contributions(round_id)
    print("Participants:", participants)
    print("Contributions:", contributions)
    
    # Verify the total contributions
    print("\nTesting total contribution retrieval...")
    total1 = server.blockchain.get_total_contribution(client_addresses[0])
    total2 = server.blockchain.get_total_contribution(client_addresses[1])
    print(f"Total contribution for client 1: {total1}")
    print(f"Total contribution for client 2: {total2}")
    
    print("\nBlockchain integration test completed successfully!")

if __name__ == "__main__":
    test_blockchain_integration()