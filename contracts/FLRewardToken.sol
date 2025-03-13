// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

contract FLRewardToken {
    string public name = "Federated Learning Token";
    string public symbol = "FLT";
    uint8 public decimals = 18;
    uint256 public totalSupply;
    
    address public owner;
    
    // Track balances for each address
    mapping(address => uint256) public balanceOf;
    
    // Track allowances (for ERC-20 compatibility)
    mapping(address => mapping(address => uint256)) public allowance;
    
    // Events
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Mint(address indexed to, uint256 value);
    
    constructor(uint256 initialSupply) {
        owner = msg.sender;
        totalSupply = initialSupply * 10**uint256(decimals);
        balanceOf[owner] = totalSupply;
        emit Transfer(address(0), owner, totalSupply);
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }
    
    // Transfer tokens
    function transfer(address to, uint256 value) public returns (bool success) {
        require(balanceOf[msg.sender] >= value, "Insufficient balance");
        
        balanceOf[msg.sender] -= value;
        balanceOf[to] += value;
        
        emit Transfer(msg.sender, to, value);
        return true;
    }
    
    // Approve spender
    function approve(address spender, uint256 value) public returns (bool success) {
        allowance[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }
    
    // Transfer from another account
    function transferFrom(address from, address to, uint256 value) public returns (bool success) {
        require(balanceOf[from] >= value, "Insufficient balance");
        require(allowance[from][msg.sender] >= value, "Insufficient allowance");
        
        balanceOf[from] -= value;
        balanceOf[to] += value;
        allowance[from][msg.sender] -= value;
        
        emit Transfer(from, to, value);
        return true;
    }
    
    // Mint new tokens (only owner)
    function mint(address to, uint256 value) public onlyOwner returns (bool success) {
        uint256 scaledValue = value * 10**uint256(decimals);
        totalSupply += scaledValue;
        balanceOf[to] += scaledValue;
        
        emit Mint(to, scaledValue);
        emit Transfer(address(0), to, scaledValue);
        return true;
    }
}