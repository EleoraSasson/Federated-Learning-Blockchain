import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
import os
from models import Net
from clients import FederatedClient
from server import FederatedServer
from utils import load_mnist_data, create_client_data
from blockchain_connector import BlockchainConnector
from web3 import Web3

def main():
    # Configuration
    num_clients = 2
    num_rounds = 1 
    client_epochs = 1
    batch_size = 64
    learning_rate = 0.01
    enable_blockchain = True  # Set to True to enable blockchain integration
    reward_per_round = 100.0  # Tokens to distribute per round
    
    # Token ledger database configuration
    token_ledger_path = "data/token_ledger.db"
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(token_ledger_path), exist_ok=True)
    
    # Load MNIST dataset
    train_dataset, test_dataset = load_mnist_data()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create client data loaders
    client_loaders, client_test_loaders, client_data_sizes = create_client_data(train_dataset, num_clients)
    
    # Initialize global model
    global_model = Net(input_dim=28*28, hidden_dim=128, output_dim=10)
    
    # Initialize blockchain connector if enabled
    blockchain = None
    w3 = None
    client_addresses = []
    
    if enable_blockchain:
        try:
            w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
            if w3.is_connected():
                print(f"Connected to Ethereum node: {w3.is_connected()}")
                print(f"Current block number: {w3.eth.block_number}")
                
                blockchain = BlockchainConnector(
                    provider_url="http://localhost:8545",
                    comm_contract_address=w3.to_checksum_address("0xe7f1725e7734ce288f8367e1bb143e90bb3f0512"),
                    contrib_contract_address=w3.to_checksum_address("0x9fe46736679d2d9a65f0992f2272de9f3c7fa6e0"),
                    reward_token_address=w3.to_checksum_address("0x809d550fca64d94Bd9F66E60752A544199cfAC3D"),
                    reward_distribution_address=w3.to_checksum_address("0x4c5859f0F772848b2D91F1D83E2Fe57935348029"),
                    comm_abi_path="blockchain/FederatedLearningCommunication.json",
                    contrib_abi_path="blockchain/FederatedLearningContribution.json",
                    reward_token_abi_path="blockchain/FLRewardToken.json",
                    reward_distribution_abi_path="blockchain/FLRewardDistribution.json"
                )
            else:
                print("Could not connect to Ethereum node. Using offline token tracking.")
                enable_blockchain = False
        except Exception as e:
            print(f"Error initializing blockchain connector: {str(e)}")
            print("Using offline token tracking instead.")
            enable_blockchain = False
    else:
        print("Blockchain integration disabled. Using offline token tracking.")
    
    # Initialize server with token ledger database
    server = FederatedServer(
        global_model=global_model,
        test_loader=test_loader,
        blockchain_enabled=enable_blockchain,
        token_ledger_path=token_ledger_path
    )
    
    if enable_blockchain and blockchain is not None:
        server.blockchain = blockchain
        print("Blockchain integration enabled")
        try:
            blockchain.debug_contract_info()
        except Exception as e:
            print(f"Error getting contract info: {str(e)}")
    
    # Define client addresses (either from blockchain or as string IDs for offline mode)
    if w3 is not None:
        # Client blockchain addresses (using Hardhat default accounts)
        client_addresses = [
            w3.to_checksum_address("0x70997970C51812dc3A010C7d01b50e0d17dc79C8"),  # Account 1
            w3.to_checksum_address("0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"),  # Account 2
            w3.to_checksum_address("0x90F79bf6EB2c4f870365E785982E1f101E93b906"),  # Account 3
            w3.to_checksum_address("0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65"),  # Account 4
            w3.to_checksum_address("0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc")   # Account 5
        ]
    else:
        # Use string identifiers for addresses in offline mode
        client_addresses = [f"client_{i}" for i in range(5)]
    
    # Initialize clients
    clients = []
    for i in range(num_clients):
        clients.append(FederatedClient(
            client_id=i,
            model=copy.deepcopy(global_model),
            data_loader=client_loaders[i],
            test_loader=test_loader,
            learning_rate=learning_rate,
            epochs=client_epochs
        ))
    
    # Map client IDs to addresses for easier lookup
    client_addresses_by_id = {}
    for i, address in enumerate(client_addresses):
        client_addresses_by_id[i] = address
        # Register the device in the token ledger database
        server.token_ledger.register_device(address, f"Client {i}")
    
    # Training history for visualization
    history = {
        'global_accuracy': [],
        'client_accuracy': {i: [] for i in range(num_clients)},
        'contributions': {i: [] for i in range(num_clients)}
    }
    
    # Track contributions by round and client
    contributions_by_round = {}
    
    # Federated learning process
    for round_idx in range(num_rounds):
        print(f"\n---- Communication Round {round_idx+1} ----")
        
        # Start a new round
        public_key = f"public_key_round_{round_idx+1}"
        server.start_round(public_key)
        
        # Clients train on their local data
        client_models = []
        for i, client in enumerate(clients):
            print(f"\nTraining Client {client.client_id}")
            client.train()
            client_models.append(copy.deepcopy(client.model))
            
            # If blockchain is enabled, submit model update
            if enable_blockchain and server.blockchain is not None:
                try:
                    model_hash = server.calculate_model_hash(client.model)
                    server.blockchain.submit_model_update(
                        round_idx+1, 
                        model_hash, 
                        client_addresses[i % len(client_addresses)]
                    )
                except Exception as e:
                    print(f"Error submitting model update: {str(e)}")
        
        # Server aggregates models and records contributions
        # This also records contributions to the token ledger database
        server.aggregate(client_models, client_data_sizes, 
                         [client_addresses[i % len(client_addresses)] for i in range(len(clients))])
        
        # Initialize round contributions tracking
        if round_idx+1 not in contributions_by_round:
            contributions_by_round[round_idx+1] = {}
        
        # Track contributions for history (already recorded in server.aggregate)
        if hasattr(server, 'contribution_scores'):
            for i, score in enumerate(server.contribution_scores):
                if i < len(clients):
                    client_id = clients[i].client_id
                    history['contributions'][client_id].append(score)
        
        # Evaluate global model
        global_accuracy = server.evaluate()
        history['global_accuracy'].append(global_accuracy)
        
        # Update clients with new global model
        global_model = server.get_global_model()
        for client in clients:
            client.update_model(global_model)
            client_acc = client.evaluate()
            history['client_accuracy'][client.client_id].append(client_acc)
        
        # Finalize round (distribute rewards) with specified reward amount
        server.finalize_round(reward_per_round)
        
        # Display reward distribution using token ledger database
        server.display_reward_distribution(round_idx+1, 
                                          [client_addresses[i % len(client_addresses)] for i in range(len(clients))])
    
    print("\nFederated Learning Completed!")
    
    # Print final results
    print("\nFinal Global Model Accuracy:", history['global_accuracy'][-1])
    print("\nFinal Client Accuracies:")
    for client_id, accuracies in history['client_accuracy'].items():
        print(f"Client {client_id}: {accuracies[-1]:.2f}%")
    
    # Display total contributions
    print("\nTotal Contributions:")
    for client_id, client_contribs in history['contributions'].items():
        if client_contribs:
            total = sum(client_contribs)
            print(f"Client {client_id}: {total:.2f}")
    
    # Display final token balances
    print("\nFinal Token Balances (from blockchain or token ledger):")
    for i, client_id in enumerate(range(num_clients)):
        if i < len(client_addresses):
            try:
                balance = server.get_token_balance(client_addresses[i % len(client_addresses)])
                print(f"Client {client_id}: {balance:.4f} tokens")
            except Exception as e:
                print(f"Error getting token balance for Client {client_id}: {str(e)}")
    
    # Display all token balances from token ledger database
    print("\nAll Token Balances (from token ledger database):")
    server.display_all_token_balances()
    
    # Export token ledger to JSON for analysis
    export_path = "data/token_ledger_export.json"
    server.export_token_ledger(export_path)
    print(f"\nToken ledger exported to {export_path}")
    
    # Display recent transaction history
    print("\nRecent Transaction History:")
    transactions = server.get_transaction_history(limit=10)
    for tx in transactions:
        print(f"Transaction {tx['tx_id'][:10]}...: {tx['device_id']} received {tx['amount']:.4f} tokens in round {tx['version']}")

if __name__ == "__main__":
    main()