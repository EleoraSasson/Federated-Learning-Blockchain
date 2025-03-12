// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FLCommunication {
    address public owner;
    
    // Events
    event FLRoundInitiated(uint256 roundId, uint256 timestamp, bytes publicKey);
    event ModelUpdate(uint256 roundId, address indexed participant, bytes modelHash);
    event GlobalModelUpdate(uint256 roundId, bytes modelHash, uint256 timestamp);
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }
    
    // Function to initiate a new federated learning round
    function initiateRound(uint256 roundId, bytes calldata publicKey) public onlyOwner {
        emit FLRoundInitiated(roundId, block.timestamp, publicKey);
    }
    
    // Function for clients to submit their model updates
    function submitModelUpdate(uint256 roundId, bytes calldata modelHash) public {
        emit ModelUpdate(roundId, msg.sender, modelHash);
    }
    
    // Function for the server to publish the new global model
    function publishGlobalModel(uint256 roundId, bytes calldata modelHash) public onlyOwner {
        emit GlobalModelUpdate(roundId, modelHash, block.timestamp);
    }
}