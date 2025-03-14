import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
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
    
    # Load MNIST dataset
    train_dataset, test_dataset = load_mnist_data()
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create client data loaders
    client_loaders, client_test_loaders, client_data_sizes = create_client_data(train_dataset, num_clients)
    
    # Initialize global model
    global_model = Net(input_dim=28*28, hidden_dim=128, output_dim=10)
    
    # Initialize blockchain connector if enabled
    blockchain = None
    if enable_blockchain:
        w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
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
    # Initialize server
    server = FederatedServer(
        global_model=global_model,
        test_loader=test_loader,
        blockchain_enabled=enable_blockchain
    )
    
    if enable_blockchain and blockchain is not None:
        server.blockchain = blockchain
        print("Blockchain integration enabled")
        blockchain.debug_contract_info()  # Add this line here

    if enable_blockchain and blockchain is not None:
        server.blockchain = blockchain
        print("Blockchain integration enabled")
    
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
    
    # Client blockchain addresses (using Hardhat default accounts)
    client_addresses = [
        w3.to_checksum_address("0x70997970C51812dc3A010C7d01b50e0d17dc79C8"),  # Account 1
        w3.to_checksum_address("0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"),  # Account 2
        w3.to_checksum_address("0x90F79bf6EB2c4f870365E785982E1f101E93b906"),  # Account 3
        w3.to_checksum_address("0x15d34AAf54267DB7D7c367839AAf71A00a2C6A65"),  # Account 4
        w3.to_checksum_address("0x9965507D1a55bcC2695C58ba16FB37d819B0A4dc")   # Account 5
    ]
    
    # Map client IDs to addresses for easier lookup
    client_addresses_by_id = {}
    for i, address in enumerate(client_addresses):
        client_addresses_by_id[i] = address
    
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
                model_hash = server.calculate_model_hash(client.model)
                server.blockchain.submit_model_update(
                    round_idx+1, 
                    model_hash, 
                    client_addresses[i % len(client_addresses)]
                )
        
        # Server aggregates models and records contributions
        server.aggregate(client_models, client_data_sizes, client_addresses)
        
        # Initialize round contributions tracking
        if round_idx+1 not in contributions_by_round:
            contributions_by_round[round_idx+1] = {}
        
        # Record contributions for each client
        if enable_blockchain and server.blockchain is not None:
            for i, client in enumerate(clients):
                client_id = client.client_id
                if i < len(client_addresses):
                    client_address = client_addresses[i]
                    try:
                        # Get contribution score
                        if hasattr(server, 'contribution_scores') and i < len(server.contribution_scores):
                            contribution_score = int(server.contribution_scores[i] * 10000)  # Scale up for blockchain
                        else:
                            # Fallback if contribution scores not available
                            contribution_score = int(20000 + (i * 1000))  # Example fallback value
                            
                        # Record on blockchain
                        server.blockchain.record_contribution(
                            client_address, 
                            round_idx+1, 
                            contribution_score
                        )
                        
                        # Store locally (scaled back down)
                        scaled_score = contribution_score / 10000
                        contributions_by_round[round_idx+1][client_address] = scaled_score
                        history['contributions'][client_id].append(scaled_score)
                        
                        print(f"Client {client_id} (address: {client_address}) contribution: {scaled_score:.2f}")
                    except Exception as e:
                        print(f"Error recording contribution for client {client_id}: {str(e)}")
        
        # Evaluate global model
        global_accuracy = server.evaluate()
        history['global_accuracy'].append(global_accuracy)
        
        # Update clients with new global model
        global_model = server.get_global_model()
        for client in clients:
            client.update_model(global_model)
            client_acc = client.evaluate()
            history['client_accuracy'][client.client_id].append(client_acc)
        
        # Finalize round on blockchain if enabled
        if enable_blockchain and server.blockchain is not None:
            try:
                server.finalize_round()
                # Display reward distribution after finalization
                try:
                    blockchain.display_reward_distribution(round_idx+1, client_addresses[:num_clients])
                except Exception as e:
                    print(f"Error displaying reward distribution: {str(e)}")
            except Exception as e:
                print(f"Error finalizing round: {str(e)}")
    
    print("\nFederated Learning with Blockchain Integration Completed!")
    
    # Print final results
    print("\nFinal Global Model Accuracy:", history['global_accuracy'][-1])
    print("\nFinal Client Accuracies:")
    for client_id, accuracies in history['client_accuracy'].items():
        print(f"Client {client_id}: {accuracies[-1]:.2f}%")
    
    if enable_blockchain:
        print("\nTotal Contributions:")
        for client_id, client_contribs in history['contributions'].items():
            if client_contribs:
                total = sum(client_contribs)
                print(f"Client {client_id}: {total:.2f}")
        
        # Display final token balances
        print("\nFinal Token Balances:")
        for i, client_id in enumerate(range(num_clients)):
            if i < len(client_addresses):
                try:
                    balance = blockchain.get_token_balance(client_addresses[i])
                    print(f"Client {client_id}: {balance:.4f} FLT")
                except Exception as e:
                    print(f"Error getting token balance for Client {client_id}: {str(e)}")
                    
        # Display total reward distribution
        print("\nTotal Rewards Distributed:")
        for round_idx in range(num_rounds):
            try:
                rewards_distributed = blockchain.reward_dist_contract.functions.rewardsDistributed(round_idx+1).call()
                if rewards_distributed:
                    reward_per_round = blockchain.reward_dist_contract.functions.rewardPerRound().call() / (10**18)
                    print(f"Round {round_idx+1}: {reward_per_round:.4f} FLT")
            except Exception as e:
                print(f"Error getting reward info for round {round_idx+1}: {str(e)}")

if __name__ == "__main__":
    main()