import sqlite3
import json
from flask import Flask, render_template, jsonify, request, send_from_directory
import os
from datetime import datetime

app = Flask(__name__)

# Configure database path
DB_PATH = os.environ.get('TOKEN_LEDGER_DB', 'data/token_ledger.db')

def get_db_connection():
    """Create a connection to the SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """Render the dashboard homepage"""
    return render_template('dashboard.html')

@app.route('/api/balances')
def get_balances():
    """Get all token balances"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT d.device_id, d.name, COALESCE(b.total_balance, 0) as balance
        FROM Devices d
        LEFT JOIN Balances b ON d.device_id = b.device_id
        ORDER BY balance DESC
    """)
    
    balances = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(balances)

@app.route('/api/contributions')
def get_contributions():
    """Get contributions by round"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT c.device_id, d.name, c.round_id, c.contribution_score
        FROM Contributions c
        JOIN Devices d ON c.device_id = d.device_id
        ORDER BY c.round_id, c.contribution_score DESC
    """)
    
    contributions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(contributions)

@app.route('/api/rounds')
def get_rounds():
    """Get distributed rounds information"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT r.round_id, r.total_reward, r.timestamp, 
               COUNT(DISTINCT c.device_id) as participant_count
        FROM DistributedRounds r
        LEFT JOIN Contributions c ON r.round_id = c.round_id
        GROUP BY r.round_id
        ORDER BY r.round_id
    """)
    
    rounds = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(rounds)

@app.route('/api/transactions')
def get_transactions():
    """Get recent transactions"""
    limit = request.args.get('limit', 50, type=int)
    device_id = request.args.get('device_id', None)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
        SELECT t.tx_id, t.device_id, d.name, t.amount, t.version as round_id, 
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
    
    cursor.execute(query, params)
    transactions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify(transactions)

@app.route('/api/device/<device_id>')
def get_device_info(device_id):
    """Get detailed information about a specific device"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get device details
    cursor.execute("""
        SELECT d.device_id, d.name, d.created_at, COALESCE(b.total_balance, 0) as balance
        FROM Devices d
        LEFT JOIN Balances b ON d.device_id = b.device_id
        WHERE d.device_id = ?
    """, (device_id,))
    
    device_row = cursor.fetchone()
    if not device_row:
        conn.close()
        return jsonify({"error": "Device not found"}), 404
        
    device = dict(device_row)
    
    # Get device contributions
    cursor.execute("""
        SELECT c.round_id, c.contribution_score, c.timestamp
        FROM Contributions c
        WHERE c.device_id = ?
        ORDER BY c.round_id
    """, (device_id,))
    
    contributions = [dict(row) for row in cursor.fetchall()]
    
    # Get device transactions
    cursor.execute("""
        SELECT t.tx_id, t.amount, t.version as round_id, t.timestamp
        FROM Transactions t
        WHERE t.device_id = ?
        ORDER BY t.timestamp DESC
    """, (device_id,))
    
    transactions = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return jsonify({
        "device": device,
        "contributions": contributions,
        "transactions": transactions
    })

@app.route('/api/stats')
def get_stats():
    """Get system-wide statistics"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    stats = {}
    
    # Total devices
    cursor.execute("SELECT COUNT(*) as count FROM Devices")
    stats["total_devices"] = cursor.fetchone()["count"]
    
    # Total tokens
    cursor.execute("SELECT SUM(total_balance) as total FROM Balances")
    result = cursor.fetchone()
    stats["total_tokens"] = result["total"] if result["total"] else 0
    
    # Total rounds
    cursor.execute("SELECT COUNT(*) as count FROM DistributedRounds")
    stats["total_rounds"] = cursor.fetchone()["count"]
    
    # Total transactions
    cursor.execute("SELECT COUNT(*) as count FROM Transactions")
    stats["total_transactions"] = cursor.fetchone()["count"]
    
    # Average contribution score
    cursor.execute("SELECT AVG(contribution_score) as avg FROM Contributions")
    result = cursor.fetchone()
    stats["avg_contribution"] = result["avg"] if result["avg"] else 0
    
    conn.close()
    
    return jsonify(stats)

@app.route('/templates/<template>')
def get_template(template):
    """Return a template file"""
    return render_template(template)

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/templates/dashboard.html')
def dashboard_template():
    """Provide the dashboard HTML template"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Federated Learning Token Dashboard</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .card {
            margin-bottom: 20px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .chart-container {
            position: relative;
            height: 300px;
            width: 100%;
        }
        .stat-card {
            text-align: center;
            padding: 15px;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .stat-label {
            color: #666;
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="#">Federated Learning Token Dashboard</a>
        </div>
    </nav>
    
    <div class="container mt-4">
        <!-- Statistics Row -->
        <div class="row mb-4" id="stats-container">
            <!-- Stats cards will be inserted here -->
        </div>
        
        <!-- Token Balances -->
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="card-title mb-0">Token Balances</h5>
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="balanceChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Contributions and Rounds -->
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-success text-white">
                        <h5 class="card-title mb-0">Contributions by Round</h5>
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="contributionsChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header bg-info text-white">
                        <h5 class="card-title mb-0">Rewards by Round</h5>
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <canvas id="roundsChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Transactions Table -->
        <div class="row">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header bg-warning text-dark">
                        <h5 class="card-title mb-0">Recent Transactions</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-striped" id="transactionsTable">
                                <thead>
                                    <tr>
                                        <th>Transaction ID</th>
                                        <th>Device</th>
                                        <th>Amount</th>
                                        <th>Round</th>
                                        <th>Timestamp</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <!-- Transaction rows will be inserted here -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Colors for charts
        const colors = [
            'rgba(54, 162, 235, 0.7)',
            'rgba(255, 99, 132, 0.7)',
            'rgba(75, 192, 192, 0.7)',
            'rgba(255, 206, 86, 0.7)',
            'rgba(153, 102, 255, 0.7)',
            'rgba(255, 159, 64, 0.7)',
            'rgba(199, 199, 199, 0.7)',
            'rgba(83, 102, 255, 0.7)',
            'rgba(40, 159, 64, 0.7)',
            'rgba(210, 99, 132, 0.7)'
        ];
        
        // Load data and create charts
        async function loadDashboard() {
            try {
                // Load statistics
                const statsResponse = await fetch('/api/stats');
                const stats = await statsResponse.json();
                displayStats(stats);
                
                // Load balances
                const balancesResponse = await fetch('/api/balances');
                const balances = await balancesResponse.json();
                createBalanceChart(balances);
                
                // Load contributions
                const contributionsResponse = await fetch('/api/contributions');
                const contributions = await contributionsResponse.json();
                createContributionsChart(contributions);
                
                // Load rounds
                const roundsResponse = await fetch('/api/rounds');
                const rounds = await roundsResponse.json();
                createRoundsChart(rounds);
                
                // Load transactions
                const transactionsResponse = await fetch('/api/transactions?limit=10');
                const transactions = await transactionsResponse.json();
                displayTransactions(transactions);
            } catch (error) {
                console.error('Error loading dashboard data:', error);
            }
        }
        
        function displayStats(stats) {
            const statsContainer = document.getElementById('stats-container');
            
            // Define stats to display
            const statCards = [
                { key: 'total_devices', label: 'Devices', color: 'primary' },
                { key: 'total_tokens', label: 'Total Tokens', color: 'success', format: value => value.toFixed(2) },
                { key: 'total_rounds', label: 'Rounds', color: 'info' },
                { key: 'total_transactions', label: 'Transactions', color: 'warning' },
                { key: 'avg_contribution', label: 'Avg Contribution', color: 'danger', format: value => value.toFixed(2) }
            ];
            
            // Generate HTML for each stat card
            statCards.forEach(card => {
                const value = stats[card.key];
                const formattedValue = card.format ? card.format(value) : value;
                
                const statCard = document.createElement('div');
                statCard.className = 'col';
                statCard.innerHTML = `
                    <div class="card stat-card border-${card.color}">
                        <div class="stat-value text-${card.color}">${formattedValue}</div>
                        <div class="stat-label">${card.label}</div>
                    </div>
                `;
                
                statsContainer.appendChild(statCard);
            });
        }
        
        function createBalanceChart(balances) {
            const ctx = document.getElementById('balanceChart').getContext('2d');
            
            // Sort by balance descending and take top 10
            const topBalances = balances
                .sort((a, b) => b.balance - a.balance)
                .slice(0, 10);
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: topBalances.map(b => truncateString(b.name || b.device_id, 10)),
                    datasets: [{
                        label: 'Token Balance',
                        data: topBalances.map(b => b.balance),
                        backgroundColor: colors,
                        borderColor: colors.map(c => c.replace('0.7', '1')),
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Tokens'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Top 10 Token Balances'
                        },
                        tooltip: {
                            callbacks: {
                                title: function(tooltipItems) {
                                    const idx = tooltipItems[0].dataIndex;
                                    return topBalances[idx].device_id;
                                }
                            }
                        }
                    }
                }
            });
        }
        
        function createContributionsChart(contributions) {
            const ctx = document.getElementById('contributionsChart').getContext('2d');
            
            // Group by round and device
            const roundsMap = {};
            const devices = new Set();
            
            contributions.forEach(c => {
                devices.add(c.device_id);
                
                if (!roundsMap[c.round_id]) {
                    roundsMap[c.round_id] = {};
                }
                
                roundsMap[c.round_id][c.device_id] = c.contribution_score;
            });
            
            const rounds = Object.keys(roundsMap).sort((a, b) => a - b);
            const devicesList = Array.from(devices);
            
            // Create datasets
            const datasets = devicesList.map((deviceId, index) => {
                const deviceData = rounds.map(round => 
                    roundsMap[round][deviceId] || 0
                );
                
                const name = contributions.find(c => c.device_id === deviceId)?.name || deviceId;
                
                return {
                    label: truncateString(name, 10),
                    data: deviceData,
                    backgroundColor: colors[index % colors.length],
                    borderColor: colors[index % colors.length].replace('0.7', '1'),
                    borderWidth: 1
                };
            });
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: rounds.map(r => `Round ${r}`),
                    datasets: datasets
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Contribution Score'
                            }
                        },
                        x: {
                            title: {
                                display: true,
                                text: 'Round'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Contributions by Device and Round'
                        }
                    }
                }
            });
        }
        
        function createRoundsChart(rounds) {
            const ctx = document.getElementById('roundsChart').getContext('2d');
            
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: rounds.map(r => `Round ${r.round_id}`),
                    datasets: [{
                        label: 'Total Reward',
                        data: rounds.map(r => r.total_reward),
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1
                    }, {
                        label: 'Participants',
                        data: rounds.map(r => r.participant_count),
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        borderColor: 'rgba(153, 102, 255, 1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Total Reward'
                            }
                        },
                        y1: {
                            beginAtZero: true,
                            position: 'right',
                            grid: {
                                drawOnChartArea: false
                            },
                            title: {
                                display: true,
                                text: 'Participants'
                            }
                        }
                    },
                    plugins: {
                        title: {
                            display: true,
                            text: 'Rewards and Participants by Round'
                        }
                    }
                }
            });
        }
        
        function displayTransactions(transactions) {
            const tableBody = document.querySelector('#transactionsTable tbody');
            
            if (transactions.length === 0) {
                const row = document.createElement('tr');
                row.innerHTML = '<td colspan="5" class="text-center">No transactions found</td>';
                tableBody.appendChild(row);
                return;
            }
            
            transactions.forEach(tx => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${tx.tx_id.substring(0, 10)}...</td>
                    <td>${tx.name} (${tx.device_id.substring(0, 8)}...)</td>
                    <td>${parseFloat(tx.amount).toFixed(4)}</td>
                    <td>${tx.round_id}</td>
                    <td>${formatDate(tx.timestamp)}</td>
                `;
                tableBody.appendChild(row);
            });
        }
        
        // Helper function to truncate strings
        function truncateString(str, maxLength) {
            if (!str) return '';
            return str.length > maxLength ? str.substring(0, maxLength) + '...' : str;
        }
        
        // Helper function to format dates
        function formatDate(dateStr) {
            if (!dateStr) return '';
            const date = new Date(dateStr);
            return date.toLocaleString();
        }
        
        // Load the dashboard when the page loads
        document.addEventListener('DOMContentLoaded', loadDashboard);
    </script>
</body>
</html>
    """

@app.route('/templates')
def list_templates():
    """List available templates"""
    return jsonify({"templates": ["dashboard.html"]})

# Create the templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Save the dashboard template
with open('templates/dashboard.html', 'w') as f:
    f.write(dashboard_template())

if __name__ == '__main__':
    print(f"Using database at: {DB_PATH}")
    if not os.path.exists(DB_PATH):
        print(f"Warning: Database file {DB_PATH} not found.")
    app.run(debug=True, host='0.0.0.0', port=5000)