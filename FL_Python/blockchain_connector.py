import json
import os
from web3 import Web3

class BlockchainConnector:
    def __init__(self, provider_url, comm_contract_address, contrib_contract_address, 
                reward_token_address, reward_distribution_address,
                comm_abi_path, contrib_abi_path, reward_token_abi_path, reward_distribution_abi_path):
        """
        Initialize the blockchain connector.
        
        Args:
            provider_url: URL of the Ethereum node (e.g. "http://localhost:8545")
            comm_contract_address: Address of the communication contract
            contrib_contract_address: Address of the contribution contract
            comm_abi_path: Path to the communication contract ABI JSON file
            contrib_abi_path: Path to the contribution contract ABI JSON file
        """
        # Connect to Ethereum node
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Ethereum node at {provider_url}")
        
        print(f"Connected to Ethereum node at {provider_url}")
        print(f"Current block number: {self.w3.eth.block_number}")
        
        # Load contract ABIs
        with open(comm_abi_path) as f:
            comm_json = json.load(f)
            comm_abi = comm_json["abi"]
        
        with open(contrib_abi_path) as f:
            contrib_json = json.load(f)
            contrib_abi = contrib_json["abi"]
        
        with open(reward_token_abi_path) as f:
            reward_token_json = json.load(f)
            reward_token_abi = reward_token_json["abi"]
        
        with open(reward_distribution_abi_path) as f:
            reward_distribution_json = json.load(f)
            reward_distribution_abi = reward_distribution_json["abi"]
        
        # Convert addresses to checksum format
        reward_token_address = self.w3.to_checksum_address(reward_token_address)
        reward_distribution_address = self.w3.to_checksum_address(reward_distribution_address)
        
        # Initialize contract instances
        self.reward_token_contract = self.w3.eth.contract(address=reward_token_address, abi=reward_token_abi)
        self.reward_dist_contract = self.w3.eth.contract(address=reward_distribution_address, abi=reward_distribution_abi)
        
        # Initialize contract instances
        self.comm_contract = self.w3.eth.contract(address=comm_contract_address, abi=comm_abi)
        self.contrib_contract = self.w3.eth.contract(address=contrib_contract_address, abi=contrib_abi)
        
        # Get an account to use for transactions
        self.accounts = self.w3.eth.accounts
        if not self.accounts:
            raise ValueError("No Ethereum accounts available")
        
        self.account = self.accounts[0]  # Use the first account
        print(f"Using Ethereum account: {self.account}")
    
    def initiate_round(self, round_id, public_key):
        """
        Initiate a new federated learning round on the blockchain.
        
        Args:
            round_id: ID of the federated learning round
            public_key: Public key for this round's encryption
            
        Returns:
            The transaction receipt
        """
        # Convert public key to bytes if it's not already
        if isinstance(public_key, str):
            public_key_bytes = public_key.encode('utf-8')
        else:
            public_key_bytes = public_key
            
        # Initiate the round on the blockchain
        tx_hash = self.comm_contract.functions.initiateRound(
            round_id, 
            public_key_bytes
        ).transact({'from': self.account})
        
        # Wait for the transaction to be mined
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Initiated round {round_id} on blockchain. Transaction hash: {tx_hash.hex()}")
        
        return tx_receipt
    
    def submit_model_update(self, round_id, model_hash, client_account=None):
        """
        Submit a model update to the blockchain.
        
        Args:
            round_id: ID of the federated learning round
            model_hash: Hash of the model update
            client_account: Account to use for the transaction (defaults to self.account)
            
        Returns:
            The transaction receipt
        """
        if client_account is None:
            client_account = self.account
            
        # Convert model hash to bytes if it's not already
        if isinstance(model_hash, str):
            model_hash_bytes = model_hash.encode('utf-8')
        else:
            model_hash_bytes = model_hash
        
        # Submit the model update
        tx_hash = self.comm_contract.functions.submitModelUpdate(
            round_id,
            model_hash_bytes
        ).transact({'from': client_account})
        
        # Wait for the transaction to be mined
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Submitted model update for round {round_id}. Transaction hash: {tx_hash.hex()}")
        
        return tx_receipt
    
    def publish_global_model(self, round_id, model_hash):
        """
        Publish the global model to the blockchain.
        
        Args:
            round_id: ID of the federated learning round
            model_hash: Hash of the global model
            
        Returns:
            The transaction receipt
        """
        # Convert model hash to bytes if it's not already
        if isinstance(model_hash, str):
            model_hash_bytes = model_hash.encode('utf-8')
        else:
            model_hash_bytes = model_hash
            
        # Publish the global model
        tx_hash = self.comm_contract.functions.publishGlobalModel(
            round_id,
            model_hash_bytes
        ).transact({'from': self.account})
        
        # Wait for the transaction to be mined
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Published global model for round {round_id}. Transaction hash: {tx_hash.hex()}")
        
        return tx_receipt
    
    def record_contribution(self, participant, round_id, contribution_score):
        """
        Record a participant's contribution on the blockchain.
        
        Args:
            participant: Address of the participant
            round_id: ID of the federated learning round
            contribution_score: Numerical score of the contribution
            
        Returns:
            The transaction receipt
        """
        # Convert to integer if needed
        if isinstance(contribution_score, float):
            # Scale to integer (e.g., multiply by 10000 to keep 4 decimal places)
            contribution_score = int(contribution_score * 10000)
        
        # Record the contribution
        tx_hash = self.contrib_contract.functions.recordContribution(
            participant,
            round_id,
            contribution_score
        ).transact({'from': self.account})
        
        # Wait for the transaction to be mined
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Recorded contribution for participant {participant} in round {round_id}. Score: {contribution_score}. Transaction hash: {tx_hash.hex()}")
        
        return tx_receipt
    
    def get_total_contribution(self, participant):
        """
        Get the total contribution of a participant across all rounds.
        
        Args:
            participant: Address of the participant
            
        Returns:
            The total contribution score
        """
        return self.contrib_contract.functions.getTotalContribution(participant).call()
    
    def get_round_contributions(self, round_id):
        """
        Get all participants' contributions for a specific round.
        
        Args:
            round_id: ID of the federated learning round
            
        Returns:
            A tuple of (participants, contributions)
        """
        return self.contrib_contract.functions.getRoundContributions(round_id).call()
    
    def get_participants(self):
        """
        Get all participants in the federated learning system.
        
        Returns:
            A list of participant addresses
        """
        return self.contrib_contract.functions.getParticipants().call()

    def distribute_rewards(self, round_id):
        """Distribute rewards for a completed round"""
        tx_hash = self.reward_dist_contract.functions.distributeRewards(
            round_id
        ).transact({'from': self.account})
            
        tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Distributed rewards for round {round_id}. Transaction hash: {tx_hash.hex()}")
            
        return tx_receipt
        
    def get_token_balance(self, address):
        """Get the token balance for an address"""
        balance = self.reward_token_contract.functions.balanceOf(address).call()
        # Convert from wei (10^18) to token units
        return balance / (10 ** 18)
        
    def get_total_rewards(self, address):
        """Get the total rewards earned by an address across all rounds"""
        balance = self.get_token_balance(address)
        return balance
            

# Example usage
if __name__ == "__main__":
    # Example of how to use the BlockchainConnector
    connector = BlockchainConnector(
        provider_url="http://localhost:8545",
        comm_contract_address="0xe7f1725e7734ce288f8367e1bb143e90bb3f0512",
        contrib_contract_address="0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0",
        comm_abi_path=r"FL_ver1\artifacts\contracts\FLCommunication.sol\FLCommunication.json",
        contrib_abi_path=r"FL_ver1\artifacts\contracts\FLCommunication.sol\FLContribution.json"
    )
    
    # Test connection
    print("Testing blockchain connector...")
    
    # Get all participants
    participants = connector.get_participants()
    print(f"Current participants: {participants}")
    
    # Example: Initiate a round
    # connector.initiate_round(1, "test_public_key")
    
    # Example: Record a contribution
    # connector.record_contribution(connector.accounts[1], 1, 95.75)