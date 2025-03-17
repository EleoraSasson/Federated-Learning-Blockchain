import sqlite3
import os
import json
from datetime import datetime

class TokenLedgerDB:
    """
    Database-backed token ledger for tracking rewards in federated learning
    """
    def __init__(self, db_path="token_ledger.db"):
        """
        Initialize the token ledger database.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Create database tables if they don't exist"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create Devices table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Devices (
                device_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # Create Transactions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Transactions (
                tx_id TEXT PRIMARY KEY,
                device_id TEXT,
                amount REAL,
                version INTEGER,
                data_cost INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hash TEXT,
                FOREIGN KEY (device_id) REFERENCES Devices(device_id)
            )
            ''')
            
            # Create Balances table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Balances (
                device_id TEXT PRIMARY KEY,
                total_balance REAL DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (device_id) REFERENCES Devices(device_id)
            )
            ''')
            
            # Create Contributions table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS Contributions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT,
                round_id INTEGER,
                contribution_score REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (device_id) REFERENCES Devices(device_id)
            )
            ''')
            
            # Create DistributedRounds table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS DistributedRounds (
                round_id INTEGER PRIMARY KEY,
                total_reward REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            conn.commit()
            print(f"Database initialized at {self.db_path}")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def register_device(self, device_id, name=None):
        """Register a device in the database"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if device exists
            cursor.execute("SELECT device_id FROM Devices WHERE device_id = ?", (device_id,))
            if cursor.fetchone() is None:
                # Insert new device
                cursor.execute(
                    "INSERT INTO Devices (device_id, name) VALUES (?, ?)",
                    (device_id, name or f"Device {device_id}")
                )
                conn.commit()
                print(f"Registered device: {device_id}")
        except sqlite3.Error as e:
            print(f"Error registering device: {e}")
        finally:
            if conn:
                conn.close()
    
    def record_contribution(self, device_id, round_id, contribution_score):
        """
        Record a device's contribution for a specific round.
        
        Args:
            device_id: ID of the device
            round_id: ID of the federated learning round
            contribution_score: Numerical score of the contribution
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Ensure device exists
            self.register_device(device_id)
            
            # Record contribution
            cursor.execute(
                "INSERT INTO Contributions (device_id, round_id, contribution_score) VALUES (?, ?, ?)",
                (device_id, round_id, contribution_score)
            )
            conn.commit()
            print(f"Recorded contribution for device {device_id} in round {round_id}: {contribution_score}")
        except sqlite3.Error as e:
            print(f"Error recording contribution: {e}")
        finally:
            if conn:
                conn.close()
    
    def distribute_rewards(self, round_id, reward_per_round=100.0):
        """
        Distribute rewards for a completed round based on relative contributions.
        
        Args:
            round_id: ID of the federated learning round
            reward_per_round: Total reward to distribute for this round
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if rewards already distributed
            cursor.execute("SELECT round_id FROM DistributedRounds WHERE round_id = ?", (round_id,))
            if cursor.fetchone() is not None:
                print(f"Rewards already distributed for round {round_id}")
                return
            
            # Get all contributions for this round
            cursor.execute(
                "SELECT device_id, contribution_score FROM Contributions WHERE round_id = ?",
                (round_id,)
            )
            contributions = cursor.fetchall()
            
            if not contributions:
                print(f"No contributions recorded for round {round_id}")
                return
            
            # Calculate total contribution
            total_contribution = sum(contrib[1] for contrib in contributions)
            
            if total_contribution <= 0:
                print(f"Total contribution for round {round_id} is zero")
                return
            
            print(f"\n--- Distributing rewards for Round {round_id} ---")
            print(f"Total reward: {reward_per_round} tokens")
            
            timestamp = datetime.now().isoformat()
            tx_id_base = f"reward_round_{round_id}_{timestamp}"
            
            # Distribute rewards proportionally
            for device_id, contribution in contributions:
                # Calculate reward share
                reward_share = (contribution / total_contribution) * reward_per_round
                
                # Create transaction record
                tx_id = f"{tx_id_base}_{device_id}"
                cursor.execute(
                    "INSERT INTO Transactions (tx_id, device_id, amount, version, data_cost, hash) VALUES (?, ?, ?, ?, ?, ?)",
                    (tx_id, device_id, reward_share, round_id, int(contribution), tx_id)
                )
                
                # Update balance
                cursor.execute(
                    """
                    INSERT INTO Balances (device_id, total_balance, last_updated)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(device_id) DO UPDATE SET
                    total_balance = total_balance + ?,
                    last_updated = CURRENT_TIMESTAMP
                    """,
                    (device_id, reward_share, reward_share)
                )
                
                print(f"Device {device_id}: Contribution {contribution:.2f} ({(contribution/total_contribution)*100:.2f}%) - Received {reward_share:.4f} tokens")
            
            # Mark round as distributed
            cursor.execute(
                "INSERT INTO DistributedRounds (round_id, total_reward) VALUES (?, ?)",
                (round_id, reward_per_round)
            )
            
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error distributing rewards: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()
    
    def get_token_balance(self, device_id):
        """
        Get the token balance for a device.
        
        Args:
            device_id: ID of the device
            
        Returns:
            The token balance
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT total_balance FROM Balances WHERE device_id = ?", (device_id,))
            result = cursor.fetchone()
            
            if result:
                return result[0]
            else:
                return 0.0
        except sqlite3.Error as e:
            print(f"Error getting token balance: {e}")
            return 0.0
        finally:
            if conn:
                conn.close()
    
    def display_reward_distribution(self, round_id, device_ids=None):
        """
        Display reward distribution information for a round.
        
        Args:
            round_id: ID of the federated learning round
            device_ids: Optional list of device IDs to display balances for
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if round has been distributed
            cursor.execute("SELECT total_reward FROM DistributedRounds WHERE round_id = ?", (round_id,))
            result = cursor.fetchone()
            
            if not result:
                print(f"Rewards for round {round_id} have not yet been distributed.")
                return
            
            reward_per_round = result[0]
            print(f"\n--- Reward Distribution for Round {round_id} ---")
            print(f"Reward per round: {reward_per_round:.4f} tokens")
            
            if device_ids is None:
                # Get all devices who received rewards in this round
                cursor.execute(
                    "SELECT DISTINCT device_id FROM Transactions WHERE version = ?", 
                    (round_id,)
                )
                device_ids = [row[0] for row in cursor.fetchall()]
            
            print("\nDevice Token Balances:")
            for device_id in device_ids:
                # Get balance
                cursor.execute("SELECT total_balance FROM Balances WHERE device_id = ?", (device_id,))
                balance_result = cursor.fetchone()
                balance = balance_result[0] if balance_result else 0.0
                
                # Get contribution for this round
                cursor.execute(
                    "SELECT contribution_score FROM Contributions WHERE device_id = ? AND round_id = ?",
                    (device_id, round_id)
                )
                contribution_result = cursor.fetchone()
                contribution = contribution_result[0] if contribution_result else 0.0
                
                print(f"Device {device_id}: {balance:.4f} tokens (Round {round_id} contribution: {contribution:.2f})")
        except sqlite3.Error as e:
            print(f"Error displaying reward distribution: {e}")
        finally:
            if conn:
                conn.close()
    
    def display_all_balances(self):
        """Display token balances for all devices"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT d.device_id, d.name, COALESCE(b.total_balance, 0) as balance
                FROM Devices d
                LEFT JOIN Balances b ON d.device_id = b.device_id
                ORDER BY balance DESC
            """)
            
            results = cursor.fetchall()
            
            print("\n--- All Token Balances ---")
            if not results:
                print("No token balances found.")
                return
                
            for device_id, name, balance in results:
                print(f"Device {device_id} ({name}): {balance:.4f} tokens")
        except sqlite3.Error as e:
            print(f"Error displaying all balances: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_transaction_history(self, device_id=None, limit=10):
        """
        Get transaction history for a device or all devices.
        
        Args:
            device_id: Optional device ID to filter by
            limit: Maximum number of transactions to return
            
        Returns:
            List of transaction records
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # Return results as dictionaries
            cursor = conn.cursor()
            
            if device_id:
                cursor.execute("""
                    SELECT tx_id, device_id, amount, version, timestamp
                    FROM Transactions
                    WHERE device_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (device_id, limit))
            else:
                cursor.execute("""
                    SELECT tx_id, device_id, amount, version, timestamp
                    FROM Transactions
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            
            transactions = [dict(row) for row in cursor.fetchall()]
            return transactions
        except sqlite3.Error as e:
            print(f"Error getting transaction history: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def export_to_json(self, file_path="token_ledger_export.json"):
        """Export the database contents to a JSON file"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get data from all tables
            data = {
                "devices": [],
                "balances": [],
                "transactions": [],
                "contributions": [],
                "distributed_rounds": []
            }
            
            # Get devices
            cursor.execute("SELECT * FROM Devices")
            data["devices"] = [dict(row) for row in cursor.fetchall()]
            
            # Get balances
            cursor.execute("SELECT * FROM Balances")
            data["balances"] = [dict(row) for row in cursor.fetchall()]
            
            # Get transactions
            cursor.execute("SELECT * FROM Transactions")
            data["transactions"] = [dict(row) for row in cursor.fetchall()]
            
            # Get contributions
            cursor.execute("SELECT * FROM Contributions")
            data["contributions"] = [dict(row) for row in cursor.fetchall()]
            
            # Get distributed rounds
            cursor.execute("SELECT * FROM DistributedRounds")
            data["distributed_rounds"] = [dict(row) for row in cursor.fetchall()]
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"Database exported to {file_path}")
            return file_path
        except (sqlite3.Error, IOError) as e:
            print(f"Error exporting database: {e}")
            return None
        finally:
            if conn:
                conn.close()


# Example usage
if __name__ == "__main__":
    # Create token ledger database
    db = TokenLedgerDB("token_ledger.db")
    
    # Register some devices
    db.register_device("0x123", "Alice's Device")
    db.register_device("0x456", "Bob's Device")
    db.register_device("0x789", "Charlie's Device")
    
    # Record contributions for a round
    db.record_contribution("0x123", 1, 75.5)
    db.record_contribution("0x456", 1, 50.2)
    db.record_contribution("0x789", 1, 90.0)
    
    # Distribute rewards
    db.distribute_rewards(1, 100.0)
    
    # Display balances
    db.display_all_balances()