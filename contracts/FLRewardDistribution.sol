// SPDX-License-Identifier: MIT
pragma solidity ^0.8.9;

import "./FLContribution.sol";  
import "./FLRewardToken.sol";   

contract FLRewardDistribution {
    FLContribution public contributionContract;
    FLRewardToken public rewardToken;
    
    address public owner;
    uint256 public rewardPerRound;
    uint256 public currentRound;
    mapping(uint256 => bool) public rewardsDistributed;
    
    event RewardsDistributed(uint256 indexed roundId, uint256 totalAmount);
    event ParticipantRewarded(address indexed participant, uint256 roundId, uint256 amount);
    
    constructor(
        address _rewardTokenAddress,
        address _contributionContractAddress,
        uint256 _rewardPerRound
    ) {
        owner = msg.sender;
        rewardToken = FLRewardToken(_rewardTokenAddress);
        contributionContract = FLContribution(_contributionContractAddress);
        rewardPerRound = _rewardPerRound;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }
    
    // Distribute rewards for a specific round
    function distributeRewards(uint256 roundId) public onlyOwner {
        require(!rewardsDistributed[roundId], "Rewards already distributed for this round");
        
        // Get all participants and their contributions for this round
        (address[] memory participants, uint256[] memory contributions) = 
            contributionContract.getRoundContributions(roundId);
        
        require(participants.length > 0, "No participants for this round");
        
        // Calculate total contribution
        uint256 totalContribution = 0;
        for (uint256 i = 0; i < contributions.length; i++) {
            totalContribution += contributions[i];
        }
        
        // Distribute rewards proportional to contributions
        for (uint256 i = 0; i < participants.length; i++) {
            if (totalContribution > 0) {
                // Calculate reward share based on contribution percentage
                uint256 rewardShare = (rewardPerRound * contributions[i]) / totalContribution;
                
                // Mint tokens to the participant
                rewardToken.mint(participants[i], rewardShare);
                
                emit ParticipantRewarded(participants[i], roundId, rewardShare);
            }
        }
        
        rewardsDistributed[roundId] = true;
        emit RewardsDistributed(roundId, rewardPerRound);
    }
    
    // Set the reward amount per round
    function setRewardPerRound(uint256 _rewardPerRound) public onlyOwner {
        rewardPerRound = _rewardPerRound;
    }
    
    // Update the contribution contract address
    function setContributionContract(address _contributionContractAddress) public onlyOwner {
        contributionContract = FLContribution(_contributionContractAddress);
    }
}