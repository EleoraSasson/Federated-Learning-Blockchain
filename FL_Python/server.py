import torch
import copy
import json
import hashlib
from blockchain_connector import BlockchainConnector
from debug_rewards import DebugRewardTracker
from token_ledger_db import TokenLedgerDB


class FederatedServer:
    def __init__(self, global_model, test_loader=None, blockchain_enabled=False, token_ledger_path="token_ledger.db"):
        """
        Initialize the federated learning server.
        
        Args:
            global_model: The initial global model
            test_loader: DataLoader for testing the global model
            blockchain_enabled: Whether to enable blockchain integration
            token_ledger_path: Path to the token ledger database
        """
        self.global_model = copy.deepcopy(global_model)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model.to(self.device)
        self.test_loader = test_loader
        self.current_round = 0
        self.blockchain_enabled = blockchain_enabled
        self.blockchain = None
        
        # Initialize token ledger database
        self.token_ledger = TokenLedgerDB(token_ledger_path)
        
        # Keep debug reward tracker for backward compatibility
        self.debug_rewards = DebugRewardTracker(reward_per_round=100.0)
        
        if blockchain_enabled:
            print("Blockchain integration enabled")
            print("Using token ledger database for offline backup")
        else:
            print("Using token ledger database for token tracking")
    
    def start_round(self, public_key=None):
        """
        Start a new federated learning round.
        
        Args:
            public_key: Optional public key for secure communication
            
        Returns:
            The round ID
        """
        self.current_round += 1
        
        # Record round initiation on blockchain if enabled
        if self.blockchain_enabled and self.blockchain is not None:
            try:
                if public_key is None:
                    public_key = f"round_{self.current_round}_key"
                
                self.blockchain.initiate_round(self.current_round, public_key)
            except Exception as e:
                print(f"Error initiating round on blockchain: {str(e)}")
                print("Continuing in offline mode")
            
        print(f"Started federated learning round {self.current_round}")
        return self.current_round
    
    def calculate_model_hash(self, model):
        """
        Calculate a hash of the model weights.
        
        Args:
            model: The PyTorch model
            
        Returns:
            A hash string representing the model
        """
        # Convert model weights to a serializable format
        model_state = {k: v.cpu().numpy().tolist() for k, v in model.state_dict().items()}
        
        # Convert to JSON string and hash
        model_json = json.dumps(str(model_state), sort_keys=True)
        model_hash = hashlib.sha256(model_json.encode()).hexdigest()
        
        return model_hash
    
    def calculate_contribution(self, global_model_before, client_model):
        """
        Calculate a client's contribution to the global model.
        
        Args:
            global_model_before: The global model before update
            client_model: The client's updated model
            
        Returns:
            A contribution score (0-100)
        """
        # Simple implementation - can be replaced with your sophisticated metric
        contribution_score = 0
        
        # Calculate magnitude of weight updates
        total_diff = 0
        total_weight = 0
        
        for (name1, param1), (name2, param2) in zip(
            global_model_before.named_parameters(), client_model.named_parameters()
        ):
            diff = torch.norm(param2.data - param1.data, p=2)
            weight = torch.norm(param1.data, p=2)
            
            total_diff += diff.item()
            total_weight += weight.item()
        
        # Normalize to a 0-100 score
        if total_weight > 0:
            contribution_score = min(100, (total_diff / total_weight) * 100)
        
        return contribution_score
    
    def record_contribution(self, participant, round_id, contribution_score):
        """
        Record a participant's contribution using blockchain or token ledger database.
        
        Args:
            participant: Address of the participant
            round_id: ID of the federated learning round
            contribution_score: Numerical score of the contribution
        """
        try:
            # Try to use blockchain if enabled
            if self.blockchain_enabled and self.blockchain is not None:
                # Convert to integer if needed
                if isinstance(contribution_score, float):
                    # Scale to integer (e.g., multiply by 10000 to keep 4 decimal places)
                    scaled_score = int(contribution_score * 10000)
                else:
                    scaled_score = contribution_score
                
                # Record on blockchain
                self.blockchain.record_contribution(
                    participant,
                    round_id,
                    scaled_score
                )
                print(f"Recorded contribution for participant {participant} in round {round_id}. Score: {contribution_score:.2f}")
            
            # Always record in token ledger database for backup/offline tracking
            self.token_ledger.record_contribution(participant, round_id, contribution_score)
            
            # Also record in debug tracker for backward compatibility
            self.debug_rewards.record_contribution(participant, round_id, contribution_score)
        except Exception as e:
            print(f"Error recording contribution on blockchain: {str(e)}")
            print("Falling back to token ledger database.")
            # Fallback to token ledger
            self.token_ledger.record_contribution(participant, round_id, contribution_score)
    
    def aggregate(self, client_models, client_data_sizes=None, client_addresses=None):
        """
        Aggregate client models and record contributions.
        
        Args:
            client_models: List of client models
            client_data_sizes: List of client dataset sizes (for weighted averaging)
            client_addresses: List of client blockchain addresses
            
        Returns:
            The updated global model
        """
        # Store the global model before aggregation for contribution calculation
        global_model_before = copy.deepcopy(self.global_model)
        
        # Perform federated averaging
        if client_data_sizes is None:
            # Equal weighting if no data sizes provided
            client_data_sizes = [1] * len(client_models)
            
        # Calculate total data size
        total_data_size = sum(client_data_sizes)
        
        # Calculate weighted average
        global_dict = self.global_model.state_dict()
        
        for k in global_dict.keys():
            global_dict[k] = torch.zeros_like(global_dict[k])
            for i, client_model in enumerate(client_models):
                weight = client_data_sizes[i] / total_data_size
                global_dict[k] += client_model.state_dict()[k] * weight
        
        # Update global model
        self.global_model.load_state_dict(global_dict)
        
        # Calculate model hash for blockchain
        global_model_hash = self.calculate_model_hash(self.global_model)
        
        # Try to publish global model update on blockchain
        if self.blockchain_enabled and self.blockchain is not None:
            try:
                self.blockchain.publish_global_model(self.current_round, global_model_hash)
            except Exception as e:
                print(f"Error publishing global model: {str(e)}")
        
        # Store contribution scores for later use
        self.contribution_scores = []
        
        # Record individual contributions if client addresses provided
        if client_addresses is not None:
            for i, (client_model, address) in enumerate(zip(client_models, client_addresses)):
                if i < len(client_models):  # Make sure we don't go out of bounds
                    # Calculate contribution
                    contribution_score = self.calculate_contribution(global_model_before, client_model)
                    self.contribution_scores.append(contribution_score)
                    
                    # Record contribution (will use blockchain or token ledger)
                    self.record_contribution(address, self.current_round, contribution_score)
                    
                    # Log
                    print(f"Client {i} (address: {address}) contribution: {contribution_score:.2f}")
        
        return self.global_model

    def evaluate(self):
        """
        Evaluate the global model on the test dataset.
        
        Returns:
            Accuracy of the global model
        """
        if self.test_loader is None:
            print("No test loader provided. Skipping evaluation.")
            return None
        
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Global Model - Test Accuracy: {accuracy:.2f}%")
        
        return accuracy

    def get_global_model(self):
        """
        Get the current global model.
        
        Returns:
            The global model
        """
        return copy.deepcopy(self.global_model)

    def finalize_round(self, reward_per_round=100.0):
        """
        Finalize the current round and distribute rewards
        
        Args:
            reward_per_round: Amount of tokens to distribute for this round
        """
        try:
            # Try to use blockchain if enabled
            if self.blockchain_enabled and self.blockchain is not None:
                # Publish the global model to the blockchain
                model_hash = self.calculate_model_hash(self.global_model)
                self.blockchain.publish_global_model(self.current_round, model_hash)
                
                # Distribute rewards using blockchain
                self.blockchain.distribute_rewards(self.current_round)
                print(f"Distributed rewards for round {self.current_round} using blockchain.")
            
            # Always distribute rewards in token ledger database for backup/offline tracking
            self.token_ledger.distribute_rewards(self.current_round, reward_per_round)
            
            # Also distribute in debug tracker for backward compatibility
            self.debug_rewards.distribute_rewards(self.current_round)
        except Exception as e:
            print(f"Error finalizing round on blockchain: {str(e)}")
            print("Falling back to token ledger database.")
            # Fallback to token ledger
            self.token_ledger.distribute_rewards(self.current_round, reward_per_round)
    
    def get_token_balance(self, address):
        """
        Get token balance using blockchain or token ledger
        
        Args:
            address: Address of the participant
            
        Returns:
            The token balance
        """
        try:
            if self.blockchain_enabled and self.blockchain is not None:
                blockchain_balance = self.blockchain.get_token_balance(address)
                # Also get token ledger balance for verification
                db_balance = self.token_ledger.get_token_balance(address)
                
                print(f"Blockchain balance: {blockchain_balance:.4f}, Database balance: {db_balance:.4f}")
                
                # Return the blockchain balance if available
                return blockchain_balance
            else:
                return self.token_ledger.get_token_balance(address)
        except Exception as e:
            print(f"Error getting token balance from blockchain: {str(e)}")
            return self.token_ledger.get_token_balance(address)
    
    def display_reward_distribution(self, round_id, client_addresses):
        """
        Display reward distribution using blockchain or token ledger
        
        Args:
            round_id: ID of the federated learning round
            client_addresses: List of client addresses
        """
        try:
            if self.blockchain_enabled and self.blockchain is not None:
                self.blockchain.display_reward_distribution(round_id, client_addresses)
            else:
                self.token_ledger.display_reward_distribution(round_id, client_addresses)
        except Exception as e:
            print(f"Error displaying reward distribution from blockchain: {str(e)}")
            self.token_ledger.display_reward_distribution(round_id, client_addresses)
    
    def display_all_token_balances(self):
        """Display all token balances using token ledger database"""
        self.token_ledger.display_all_balances()
    
    def get_transaction_history(self, address=None, limit=10):
        """
        Get transaction history for a participant or all participants
        
        Args:
            address: Optional address to filter by
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction records
        """
        return self.token_ledger.get_transaction_history(address, limit)
    
    def export_token_ledger(self, file_path="token_ledger_export.json"):
        """
        Export token ledger data to a JSON file
        
        Args:
            file_path: Path to save the export
            
        Returns:
            Path to the exported file
        """
        return self.token_ledger.export_to_json(file_path)