import os
import sqlite3
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

class TokenLedgerInteractive:
    def __init__(self, root, db_path="data/token_ledger.db"):
        """Initialize the interactive token ledger viewer"""
        self.root = root
        self.db_path = db_path
        
        # Configure the root window
        root.title("Federated Learning Token Ledger")
        root.geometry("900x600")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a notebook (tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_devices_tab()
        self.create_balances_tab()
        self.create_transactions_tab()
        self.create_contributions_tab()
        self.create_rounds_tab()
        
        # Create status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create menu
        self.create_menu()
        
        # Load initial data
        self.load_data()
    
    def create_menu(self):
        """Create the application menu"""
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Database...", command=self.open_database)
        file_menu.add_command(label="Export to JSON...", command=self.export_to_json)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Data menu
        data_menu = tk.Menu(menubar, tearoff=0)
        data_menu.add_command(label="Refresh All", command=self.load_data)
        menubar.add_cascade(label="Data", menu=data_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.root.config(menu=menubar)
    
    def create_devices_tab(self):
        """Create the devices tab"""
        devices_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(devices_frame, text="Devices")
        
        # Create treeview for devices
        columns = ("device_id", "name", "created_at", "balance", "transactions")
        self.devices_tree = ttk.Treeview(devices_frame, columns=columns, show="headings")
        
        # Configure columns
        self.devices_tree.heading("device_id", text="Device ID")
        self.devices_tree.heading("name", text="Name")
        self.devices_tree.heading("created_at", text="Created")
        self.devices_tree.heading("balance", text="Balance")
        self.devices_tree.heading("transactions", text="Transactions")
        
        self.devices_tree.column("device_id", width=200)
        self.devices_tree.column("name", width=150)
        self.devices_tree.column("created_at", width=150)
        self.devices_tree.column("balance", width=100)
        self.devices_tree.column("transactions", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(devices_frame, orient=tk.VERTICAL, command=self.devices_tree.yview)
        self.devices_tree.configure(yscroll=scrollbar.set)
        
        # Pack widgets
        self.devices_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_balances_tab(self):
        """Create the balances tab"""
        balances_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(balances_frame, text="Token Balances")
        
        # Create top frame for summary
        top_frame = ttk.Frame(balances_frame)
        top_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Total tokens label
        self.total_tokens_var = tk.StringVar()
        ttk.Label(top_frame, text="Total Tokens:").pack(side=tk.LEFT)
        ttk.Label(top_frame, textvariable=self.total_tokens_var, font=("TkDefaultFont", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Create treeview for balances
        columns = ("device_id", "name", "balance", "last_updated")
        self.balances_tree = ttk.Treeview(balances_frame, columns=columns, show="headings")
        
        # Configure columns
        self.balances_tree.heading("device_id", text="Device ID")
        self.balances_tree.heading("name", text="Name")
        self.balances_tree.heading("balance", text="Balance")
        self.balances_tree.heading("last_updated", text="Last Updated")
        
        self.balances_tree.column("device_id", width=200)
        self.balances_tree.column("name", width=150)
        self.balances_tree.column("balance", width=100)
        self.balances_tree.column("last_updated", width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(balances_frame, orient=tk.VERTICAL, command=self.balances_tree.yview)
        self.balances_tree.configure(yscroll=scrollbar.set)
        
        # Pack widgets
        self.balances_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_transactions_tab(self):
        """Create the transactions tab"""
        transactions_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(transactions_frame, text="Transactions")
        
        # Create controls frame
        controls_frame = ttk.Frame(transactions_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Device filter
        ttk.Label(controls_frame, text="Device:").pack(side=tk.LEFT, padx=(0, 5))
        self.device_filter_var = tk.StringVar()
        self.device_combo = ttk.Combobox(controls_frame, textvariable=self.device_filter_var, width=30)
        self.device_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Limit filter
        ttk.Label(controls_frame, text="Limit:").pack(side=tk.LEFT, padx=(0, 5))
        self.limit_var = tk.StringVar(value="50")
        limit_entry = ttk.Entry(controls_frame, textvariable=self.limit_var, width=5)
        limit_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Apply filter button
        ttk.Button(controls_frame, text="Apply Filter", command=self.load_transactions).pack(side=tk.LEFT)
        
        # Create treeview for transactions
        columns = ("tx_id", "device", "amount", "round", "timestamp", "data_cost")
        self.transactions_tree = ttk.Treeview(transactions_frame, columns=columns, show="headings")
        
        # Configure columns
        self.transactions_tree.heading("tx_id", text="Transaction ID")
        self.transactions_tree.heading("device", text="Device")
        self.transactions_tree.heading("amount", text="Amount")
        self.transactions_tree.heading("round", text="Round")
        self.transactions_tree.heading("timestamp", text="Timestamp")
        self.transactions_tree.heading("data_cost", text="Data Cost")
        
        self.transactions_tree.column("tx_id", width=200)
        self.transactions_tree.column("device", width=150)
        self.transactions_tree.column("amount", width=100)
        self.transactions_tree.column("round", width=80)
        self.transactions_tree.column("timestamp", width=150)
        self.transactions_tree.column("data_cost", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(transactions_frame, orient=tk.VERTICAL, command=self.transactions_tree.yview)
        self.transactions_tree.configure(yscroll=scrollbar.set)
        
        # Pack widgets
        self.transactions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_contributions_tab(self):
        """Create the contributions tab"""
        contributions_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(contributions_frame, text="Contributions")
        
        # Create controls frame
        controls_frame = ttk.Frame(contributions_frame)
        controls_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Round filter
        ttk.Label(controls_frame, text="Round:").pack(side=tk.LEFT, padx=(0, 5))
        self.round_filter_var = tk.StringVar(value="All")
        self.round_combo = ttk.Combobox(contributions_frame, textvariable=self.round_filter_var, width=10)
        self.round_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Apply filter button
        ttk.Button(controls_frame, text="Apply Filter", command=self.load_contributions).pack(side=tk.LEFT)
        
        # Create treeview for contributions
        columns = ("device_id", "name", "round", "contribution_score", "timestamp")
        self.contributions_tree = ttk.Treeview(contributions_frame, columns=columns, show="headings")
        
        # Configure columns
        self.contributions_tree.heading("device_id", text="Device ID")
        self.contributions_tree.heading("name", text="Name")
        self.contributions_tree.heading("round", text="Round")
        self.contributions_tree.heading("contribution_score", text="Contribution Score")
        self.contributions_tree.heading("timestamp", text="Timestamp")
        
        self.contributions_tree.column("device_id", width=200)
        self.contributions_tree.column("name", width=150)
        self.contributions_tree.column("round", width=80)
        self.contributions_tree.column("contribution_score", width=150)
        self.contributions_tree.column("timestamp", width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(contributions_frame, orient=tk.VERTICAL, command=self.contributions_tree.yview)
        self.contributions_tree.configure(yscroll=scrollbar.set)
        
        # Pack widgets
        self.contributions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_rounds_tab(self):
        """Create the rounds tab"""
        rounds_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(rounds_frame, text="Rounds")
        
        # Create treeview for rounds
        columns = ("round_id", "total_reward", "timestamp", "participant_count")
        self.rounds_tree = ttk.Treeview(rounds_frame, columns=columns, show="headings")
        
        # Configure columns
        self.rounds_tree.heading("round_id", text="Round ID")
        self.rounds_tree.heading("total_reward", text="Total Reward")
        self.rounds_tree.heading("timestamp", text="Timestamp")
        self.rounds_tree.heading("participant_count", text="Participants")
        
        self.rounds_tree.column("round_id", width=100)
        self.rounds_tree.column("total_reward", width=150)
        self.rounds_tree.column("timestamp", width=150)
        self.rounds_tree.column("participant_count", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(rounds_frame, orient=tk.VERTICAL, command=self.rounds_tree.yview)
        self.rounds_tree.configure(yscroll=scrollbar.set)
        
        # Pack widgets
        self.rounds_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
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
            self.show_error(f"Database error: {e}")
            return []
        finally:
            if conn:
                conn.close()
    
    def load_data(self):
        """Load all data from the database"""
        if not os.path.exists(self.db_path):
            self.show_error(f"Database file not found: {self.db_path}")
            return
        
        try:
            self.load_devices()
            self.load_balances()
            self.load_transactions()
            self.load_contributions()
            self.load_rounds()
            self.update_device_combos()
            self.update_round_combos()
            
            self.status_var.set(f"Data loaded from {self.db_path}")
        except Exception as e:
            self.show_error(f"Error loading data: {e}")
    
    def load_devices(self):
        """Load devices data into the devices tab"""
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
        
        # Clear existing data
        for item in self.devices_tree.get_children():
            self.devices_tree.delete(item)
        
        # Insert new data
        for dev in devices:
            self.devices_tree.insert("", tk.END, values=(
                dev["device_id"],
                dev["name"],
                dev["created_at"],
                f"{dev['balance']:.4f}",
                dev["num_transactions"]
            ))
    
    def load_balances(self):
        """Load balance data into the balances tab"""
        balances = self.execute_query("""
            SELECT d.device_id, d.name, COALESCE(b.total_balance, 0) as balance, 
                   b.last_updated
            FROM Devices d
            LEFT JOIN Balances b ON d.device_id = b.device_id
            ORDER BY balance DESC
        """)
        
        # Clear existing data
        for item in self.balances_tree.get_children():
            self.balances_tree.delete(item)
        
        # Calculate total tokens
        total_tokens = sum(bal["balance"] for bal in balances)
        self.total_tokens_var.set(f"{total_tokens:.4f}")
        
        # Insert new data
        for bal in balances:
            self.balances_tree.insert("", tk.END, values=(
                bal["device_id"],
                bal["name"],
                f"{bal['balance']:.4f}",
                bal["last_updated"] or "Never"
            ))
    
    def load_transactions(self):
        """Load transaction data into the transactions tab"""
        # Get filter values
        device_id = self.device_filter_var.get() if self.device_filter_var.get() != "All" else None
        try:
            limit = int(self.limit_var.get())
        except ValueError:
            limit = 50
        
        # Build query
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
        
        # Clear existing data
        for item in self.transactions_tree.get_children():
            self.transactions_tree.delete(item)
        
        # Insert new data
        for tx in transactions:
            self.transactions_tree.insert("", tk.END, values=(
                tx["tx_id"],
                f"{tx['name']} ({tx['device_id'][:8]}...)",
                f"{tx['amount']:.4f}",
                tx["round"],
                tx["timestamp"],
                tx["data_cost"]
            ))
    
    def load_contributions(self):
        """Load contribution data into the contributions tab"""
        # Get filter values
        round_id = self.round_filter_var.get() if self.round_filter_var.get() != "All" else None
        
        # Build query
        if round_id and round_id.isdigit():
            query = """
                SELECT c.device_id, d.name, c.round_id, c.contribution_score, c.timestamp
                FROM Contributions c
                JOIN Devices d ON c.device_id = d.device_id
                WHERE c.round_id = ?
                ORDER BY c.contribution_score DESC
            """
            contributions = self.execute_query(query, (int(round_id),))
        else:
            query = """
                SELECT c.device_id, d.name, c.round_id, c.contribution_score, c.timestamp
                FROM Contributions c
                JOIN Devices d ON c.device_id = d.device_id
                ORDER BY c.round_id, c.contribution_score DESC
            """
            contributions = self.execute_query(query)
        
        # Clear existing data
        for item in self.contributions_tree.get_children():
            self.contributions_tree.delete(item)
        
        # Insert new data
        for con in contributions:
            self.contributions_tree.insert("", tk.END, values=(
                con["device_id"],
                con["name"],
                con["round_id"],
                f"{con['contribution_score']:.2f}",
                con["timestamp"]
            ))
    
    def load_rounds(self):
        """Load rounds data into the rounds tab"""
        rounds = self.execute_query("""
            SELECT r.round_id, r.total_reward, r.timestamp, 
                   COUNT(DISTINCT c.device_id) as participant_count
            FROM DistributedRounds r
            LEFT JOIN Contributions c ON r.round_id = c.round_id
            GROUP BY r.round_id
            ORDER BY r.round_id
        """)
        
        # Clear existing data
        for item in self.rounds_tree.get_children():
            self.rounds_tree.delete(item)
        
        # Insert new data
        for r in rounds:
            self.rounds_tree.insert("", tk.END, values=(
                r["round_id"],
                f"{r['total_reward']:.4f}",
                r["timestamp"],
                r["participant_count"]
            ))
    
    def update_device_combos(self):
        """Update device filter comboboxes"""
        devices = self.execute_query("SELECT device_id, name FROM Devices ORDER BY name")
        
        # Prepare values for combobox
        device_values = ["All"] + [f"{dev['device_id']}" for dev in devices]
        
        # Update combobox values
        self.device_combo['values'] = device_values
        if self.device_filter_var.get() not in device_values:
            self.device_filter_var.set("All")
    
    def update_round_combos(self):
        """Update round filter comboboxes"""
        rounds = self.execute_query("SELECT DISTINCT round_id FROM Contributions ORDER BY round_id")
        
        # Prepare values for combobox
        round_values = ["All"] + [str(r['round_id']) for r in rounds]
        
        # Update combobox values
        self.round_combo['values'] = round_values
        if self.round_filter_var.get() not in round_values:
            self.round_filter_var.set("All")
    
    def open_database(self):
        """Open a different database file"""
        file_path = filedialog.askopenfilename(
            title="Open Token Ledger Database",
            filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.db_path = file_path
            self.load_data()
    
    def export_to_json(self):
        """Export database to JSON format"""
        file_path = filedialog.asksaveasfilename(
            title="Export to JSON",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return
        
        data = {
            "devices": self.execute_query("SELECT * FROM Devices"),
            "balances": self.execute_query("SELECT * FROM Balances"),
            "transactions": self.execute_query("SELECT * FROM Transactions"),
            "contributions": self.execute_query("SELECT * FROM Contributions"),
            "distributed_rounds": self.execute_query("SELECT * FROM DistributedRounds"),
            "export_time": datetime.now().isoformat()
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.status_var.set(f"Data exported to {file_path}")
            messagebox.showinfo("Export Complete", f"Data exported to {file_path}")
        except Exception as e:
            self.show_error(f"Error exporting data: {e}")
    
    def show_error(self, message):
        """Show error message"""
        self.status_var.set(f"Error: {message}")
        messagebox.showerror("Error", message)
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Federated Learning Token Ledger Viewer\n\n"
            "A tool for viewing and analyzing token distribution "
            "in federated learning systems."
        )

def main():
    # Create the root window
    root = tk.Tk()
    
    # Determine the default database path
    default_db_path = "data/token_ledger.db"
    if not os.path.exists(default_db_path):
        # Check current directory
        if os.path.exists("token_ledger.db"):
            default_db_path = "token_ledger.db"
        # Allow user to specify path if not found
        else:
            file_path = filedialog.askopenfilename(
                title="Open Token Ledger Database",
                filetypes=[("SQLite Database", "*.db"), ("All Files", "*.*")]
            )
            if file_path:
                default_db_path = file_path
    
    # Create the application
    app = TokenLedgerInteractive(root, default_db_path)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main()