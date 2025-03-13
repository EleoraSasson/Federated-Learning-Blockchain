# Federated Learning with Blockchain Integration

A federated learning system integrated with blockchain for transparency, accountability, and incentivization.

## Overview
Federated Learning enables collaborative model training without sharing raw data. Blockchain ensures verifiable contribution tracking and immutable record-keeping.

### Features
- Decentralized model training with federated averaging
- Transparent tracking via smart contracts
- Immutable model updates and contributions
- Reward mechanisms based on contribution metrics

## Architecture

### Federated Learning System
- **Server**: Aggregates model updates and tracks contributions
- **Clients**: Train models locally and submit updates
- **Models**: Neural networks for the learning task

### Blockchain Integration
- **Smart Contracts**: Store and verify contributions
- **Blockchain Connector**: Interfaces with Ethereum

## Tech Stack
- **Python**, **PyTorch** (model training)
- **Ethereum**, **Solidity**, **Hardhat** (smart contracts)
- **Web3.py** (blockchain interaction)

## Smart Contracts
1. **FLCommunication**: Manages federated learning rounds
2. **FLContribution**: Records and tracks contributions

## Setup

### Prerequisites
- Python 3.8+, PyTorch, Node.js, npm, Hardhat, web3.py

### Installation
```bash
# Clone repository
git clone https://github.com/EleoraSasson/Federated-Learning-Blockchain.git
cd federated-learning-blockchain

# Install dependencies
pip install -r requirements.txt
cd federated-contracts
npm install

# Compile smart contracts
npx hardhat compile
```

### Running the System
```bash
# Start Ethereum node
npx hardhat node

# Deploy smart contracts
npx hardhat run scripts/deploy.js --network localhost

# Update contract addresses in main.py

# Run federated learning system
python main.py
```

## Project Structure
```
federated-learning-blockchain/
├── federated-contracts/        # Smart contracts
│   ├── contracts/              # FLCommunication.sol, FLContribution.sol
│   ├── scripts/                # Deployment scripts
│   └── test/                   # Contract tests
├── FL_Python/                  # Python implementation
│   ├── blockchain/             # ABI & blockchain connector
│   ├── main.py                 # Execution script
│   ├── clients.py, server.py   # FL components
│   ├── models.py               # Neural network models
│   └── utils.py                # Utility functions
├── .gitignore
└── README.md
```

## Contribution Calculation
- Evaluates magnitude, impact, and consistency of model updates
- Recorded on-chain for incentivization and rewards

## Results (MNIST Dataset)
- Accuracy improvement: **93% → 98%** (10 rounds)
- Effective contribution tracking via blockchain


