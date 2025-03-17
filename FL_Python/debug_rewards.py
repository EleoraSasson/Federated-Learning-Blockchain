import json
import os

class DebugRewardTracker:
    """
    A class to track token rewards without using blockchain contracts.
    This is useful for debugging or when blockchain integration isn't working.
    """
    def __init__(self, reward_per_round=100.0, save_file="debug_rewards.json"):
        """
        Initialize the reward tracker.
        
        Args:
            reward_per_round: Amount of tokens to distribute each round
            save_file: Path to save the token balance data
        """
        self.reward_per_round = reward_per_round
        self.save_file = save_file
        self.token_balances = {}
        self.contributions = {}
        self.distributed_rounds = set()
        
        # Load existing data if available
        self.load_data()
    
    def load_data(self):
        """Load token balances and contribution data from file if it exists"""
        if os.path.exists(self.save_file):
            try:
                with open(self.save_file, 'r') as f:
                    data = json.load(f)
                    self.token_balances = data.get('token_balances', {})
                    self.contributions = data.get('contributions', {})
                    self.distributed_rounds = set(data.get('distributed_rounds', []))
                print(f"Loaded reward data from {self.save_file}")
            except Exception as e:
                print(f"Error loading reward data: {str(e)}")
    
    def save_data(self):
        """Save token balances and contribution data to file"""
        try:
            data = {
                'token_balances': self.token_balances,
                'contributions': self.contributions,
                'distributed_rounds': list(self.distributed_rounds)
            }
            with open(self.save_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved reward data to {self.save_file}")
        except Exception as e:
            print(f"Error saving reward data: {str(e)}")
    
    def record_contribution(self, participant, round_id, contribution_score):
        """
        Record a participant's contribution for a specific round.
        
        Args:
            participant: Address or ID of the participant
            round_id: ID of the federated learning round
            contribution_score: Numerical score of the contribution
        """
        # Convert to string for JSON serialization
        participant = str(participant)
        round_id = str(round_id)
        
        # Initialize data structures if needed
        if round_id not in self.contributions:
            self.contributions[round_id] = {}
        
        # Record contribution
        self.contributions[round_id][participant] = float(contribution_score)
        
        print(f"Recorded contribution for participant {participant} in round {round_id}: {contribution_score}")
        self.save_data()
    
    def distribute_rewards(self, round_id):
        """
        Distribute rewards for a completed round based on relative contributions.
        
        Args:
            round_id: ID of the federated learning round
        """
        round_id = str(round_id)
        
        # Check if rewards already distributed
        if round_id in self.distributed_rounds:
            print(f"Rewards already distributed for round {round_id}")
            return
        
        # Check if there are any contributions for this round
        if round_id not in self.contributions or not self.contributions[round_id]:
            print(f"No contributions recorded for round {round_id}")
            return
        
        # Calculate total contribution for the round
        round_contributions = self.contributions[round_id]
        total_contribution = sum(round_contributions.values())
        
        if total_contribution <= 0:
            print(f"Total contribution for round {round_id} is zero")
            return
        
        print(f"\n--- Distributing rewards for Round {round_id} ---")
        print(f"Total reward: {self.reward_per_round} tokens")
        
        # Distribute rewards proportionally to contribution
        for participant, contribution in round_contributions.items():
            # Calculate reward share
            reward_share = (contribution / total_contribution) * self.reward_per_round
            
            # Update balance
            if participant not in self.token_balances:
                self.token_balances[participant] = 0.0
            
            self.token_balances[participant] += reward_share
            
            print(f"Participant {participant}: Contribution {contribution:.2f} ({(contribution/total_contribution)*100:.2f}%) - Received {reward_share:.4f} tokens")
        
        # Mark round as distributed
        self.distributed_rounds.add(round_id)
        
        self.save_data()
    
    def get_token_balance(self, participant):
        """
        Get the token balance for a participant.
        
        Args:
            participant: Address or ID of the participant
            
        Returns:
            The token balance
        """
        participant = str(participant)
        return self.token_balances.get(participant, 0.0)
    
    def display_reward_distribution(self, round_id, participants=None):
        """
        Display reward distribution information for a round.
        
        Args:
            round_id: ID of the federated learning round
            participants: Optional list of participants to display balances for
        """
        round_id = str(round_id)
        
        if round_id not in self.distributed_rounds:
            print(f"Rewards for round {round_id} have not yet been distributed.")
            return
        
        print(f"\n--- Reward Distribution for Round {round_id} ---")
        print(f"Reward per round: {self.reward_per_round:.4f} tokens")
        
        if participants is None:
            # Display all participants who received rewards in this round
            participants = list(self.contributions.get(round_id, {}).keys())
        
        print("\nParticipant Token Balances:")
        for participant in participants:
            participant = str(participant)
            balance = self.get_token_balance(participant)
            contribution = self.contributions.get(round_id, {}).get(participant, 0.0)
            print(f"Participant {participant}: {balance:.4f} tokens (Round {round_id} contribution: {contribution:.2f})")
    
    def display_all_balances(self):
        """Display token balances for all participants"""
        print("\n--- All Token Balances ---")
        for participant, balance in sorted(self.token_balances.items(), key=lambda x: x[1], reverse=True):
            print(f"Participant {participant}: {balance:.4f} tokens")


# Example usage
if __name__ == "__main__":
    # Create reward tracker
    tracker = DebugRewardTracker(reward_per_round=100.0)
    
    # Record contributions for a round
    tracker.record_contribution("0x123", 1, 75.5)
    tracker.record_contribution("0x456", 1, 50.2)
    tracker.record_contribution("0x789", 1, 90.0)
    
    # Distribute rewards
    tracker.distribute_rewards(1)
    
    # Display balances
    tracker.display_all_balances()