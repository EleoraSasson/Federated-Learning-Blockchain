import argparse
import sqlite3
import json
import os
from tabulate import tabulate  # pip install tabulate
from datetime import datetime

class TokenLedgerViewer:
    def __init__(self, db_path):
        """Initialize the token ledger viewer"""
        self.db_path = db_path
        if not os.path.exists(db_path):
            print(f"Error: Database file '{db_path}' not found.")
            exit(1)
    
    def execute_query(self, query, params=None):
        """Execute a SQL query and return results"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def show_devices(self):
        """Show all registered devices"""
        devices = self.execute_query("""
            SELECT d.device_id, d.name, d.created_at, 
                   COALESCE(b.total_balance, 0) as balance,
                   COUNT(t.tx_id) as num_transactions
            FROM Devices d
            LEFT JOIN Balances b ON d.device_id = b.device_id
            LEFT JOIN Transactions t ON d.device_id = t.device_id
            GROUP BY d.device_id
            ORDER BY balance DESC
        """)
        
        if not devices:
            print("No devices found.")
            return
        
        print("\n=== Registered Devices ===\n")
        headers = ["Device ID", "Name", "Created", "Balance", "Transactions"]
        rows = [[
            dev["device_id"], 
            dev["name"], 
            dev["created_at"], 
            f"{dev['balance']:.4f}", 
            dev["num_transactions"]
        ] for dev in devices]
        
        print(tabulate(rows, headers=headers, tablefmt="pretty"))
        print(f"\nTotal Devices: {len(devices)}")
    
    def show_balances(self):
        """Show token balances for all devices"""
        balances = self.execute_query("""
            SELECT d.device_id, d.name, COALESCE(b.total_balance, 0) as balance, 
                   b.last_updated
            FROM Devices d
            LEFT JOIN Balances b ON d.device_id = b.device_id
            ORDER BY balance DESC
        """)
        
        if not balances:
            print("No balances found.")
            return
        
        print("\n=== Token Balances ===\n")
        headers = ["Device ID", "Name", "Balance", "Last Updated"]
        rows = [[
            bal["device_id"], 
            bal["name"], 
            f"{bal['balance']:.4f}", 
            bal["last_updated"] or "Never"
        ] for bal in balances]
        
        print(tabulate(rows, headers=headers, tablefmt="pretty"))
        
        # Calculate total tokens distributed
        total_tokens = sum(bal["balance"] for bal in balances)
        print(f"\nTotal Tokens Distributed: {total_tokens:.4f}")
    
    def show_transactions(self, limit=20, device_id=None):
        """Show recent transactions"""
        query = """
            SELECT t.tx_id, t.device_id, d.name, t.amount, t.version as round, 
                   t.timestamp, t.data_cost
            FROM Transactions t
            JOIN Devices d ON t.device_id = d.device_id
        """
        params = []
        
        if device_id:
            query += " WHERE t.device_id = ?"
            params.append(device_id)
        
        query += " ORDER BY t.timestamp DESC LIMIT ?"
        params.append(limit)
        
        transactions = self.execute_query(query, params)
        
        if not transactions:
            print("No transactions found.")
            return
        
        print(f"\n=== Recent Transactions {'for ' + device_id if device_id else ''} ===\n")
        headers = ["Transaction ID", "Device", "Amount", "Round", "Timestamp", "Data Cost"]
        rows = [[
            tx["tx_id"][:10] + "...", 
            f"{tx['device_id']} ({tx['name']})", 
            f"{tx['amount']:.4f}", 
            tx["round"], 
            tx["timestamp"], 
            tx["data_cost"]
        ] for tx in transactions]
        
        print(tabulate(rows, headers=headers, tablefmt="pretty"))
    
    def show_contributions(self, round_id=None):
        """Show contributions by round"""
        if round_id:
            query = """
                SELECT c.device_id, d.name, c.round_id, c.contribution_score, c.timestamp
                FROM Contributions c
                JOIN Devices d ON c.device_id = d.device_id
                WHERE c.round_id = ?
                ORDER BY c.contribution_score DESC
            """
            contributions = self.execute_query(query, (round_id,))
            
            if not contributions:
                print(f"No contributions found for round {round_id}.")
                return
            
            print(f"\n=== Contributions for Round {round_id} ===\n")
        else:
            query = """
                SELECT c.device_id, d.name, c.round_id, c.contribution_score, c.timestamp
                FROM Contributions c
                JOIN Devices d ON c.device_id = d.device_id
                ORDER BY c.round_id, c.contribution_score DESC
            """
            contributions = self.execute_query(query)
            
            if not contributions:
                print("No contributions found.")
                return
            
            print("\n=== All Contributions by Round ===\n")
        
        headers = ["Device ID", "Name", "Round", "Contribution Score", "Timestamp"]
        rows = [[
            con["device_id"], 
            con["name"], 
            con["round_id"], 
            f"{con['contribution_score']:.2f}", 
            con["timestamp"]
        ] for con in contributions]
        
        print(tabulate(rows, headers=headers, tablefmt="pretty"))
        
        # Calculate statistics by round
        round_stats = {}
        for con in contributions:
            round_id = con["round_id"]
            if round_id not in round_stats:
                round_stats[round_id] = {
                    "count": 0,
                    "total": 0,
                    "max": 0,
                    "min": float('inf')
                }
            
            score = con["contribution_score"]
            round_stats[round_id]["count"] += 1
            round_stats[round_id]["total"] += score
            round_stats[round_id]["max"] = max(round_stats[round_id]["max"], score)
            round_stats[round_id]["min"] = min(round_stats[round_id]["min"], score)
        
        print("\n=== Contribution Statistics by Round ===\n")
        stats_headers = ["Round", "Participants", "Total Score", "Average", "Max", "Min"]
        stats_rows = []
        
        for r_id, stats in sorted(round_stats.items()):
            if stats["count"] > 0:
                avg = stats["total"] / stats["count"]
                stats_rows.append([
                    r_id,
                    stats["count"],
                    f"{stats['total']:.2f}",
                    f"{avg:.2f}",
                    f"{stats['max']:.2f}",
                    f"{stats['min']:.2f}" if stats["min"] != float('inf') else "N/A"
                ])
        
        print(tabulate(stats_rows, headers=stats_headers, tablefmt="pretty"))
    
    def show_rounds(self):
        """Show distributed rounds information"""
        rounds = self.execute_query("""
            SELECT r.round_id, r.total_reward, r.timestamp, 
                   COUNT(DISTINCT c.device_id) as participant_count
            FROM DistributedRounds r
            LEFT JOIN Contributions c ON r.round_id = c.round_id
            GROUP BY r.round_id
            ORDER BY r.round_id
        """)
        
        if not rounds:
            print("No distributed rounds found.")
            return
        
        print("\n=== Distributed Rounds ===\n")
        headers = ["Round ID", "Total Reward", "Timestamp", "Participants"]
        rows = [[
            r["round_id"], 
            f"{r['total_reward']:.4f}", 
            r["timestamp"], 
            r["participant_count"]
        ] for r in rounds]
        
        print(tabulate(rows, headers=headers, tablefmt="pretty"))
    
    def export_to_json(self, output_path):
        """Export database to JSON format"""
        data = {
            "devices": self.execute_query("SELECT * FROM Devices"),
            "balances": self.execute_query("SELECT * FROM Balances"),
            "transactions": self.execute_query("SELECT * FROM Transactions"),
            "contributions": self.execute_query("SELECT * FROM Contributions"),
            "distributed_rounds": self.execute_query("SELECT * FROM DistributedRounds"),
            "export_time": datetime.now().isoformat()
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"Database exported to {output_path}")
            return True
        except Exception as e:
            print(f"Error exporting database: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Token Ledger Database Viewer')
    parser.add_argument('--db', type=str, default='data/token_ledger.db',
                        help='Path to the token ledger database')
    parser.add_argument('--action', type=str, required=True,
                        choices=['devices', 'balances', 'transactions', 'contributions', 'rounds', 'export'],
                        help='Action to perform')
    parser.add_argument('--device', type=str, help='Filter by device ID')
    parser.add_argument('--round', type=int, help='Filter by round ID')
    parser.add_argument('--limit', type=int, default=20, help='Limit number of results')
    parser.add_argument('--output', type=str, default='token_ledger_export.json',
                        help='Output path for export')
    
    args = parser.parse_args()
    viewer = TokenLedgerViewer(args.db)
    
    if args.action == 'devices':
        viewer.show_devices()
    elif args.action == 'balances':
        viewer.show_balances()
    elif args.action == 'transactions':
        viewer.show_transactions(args.limit, args.device)
    elif args.action == 'contributions':
        viewer.show_contributions(args.round)
    elif args.action == 'rounds':
        viewer.show_rounds()
    elif args.action == 'export':
        viewer.export_to_json(args.output)

if __name__ == "__main__":
    main()