// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FLContribution {
    address public owner;
    
    // Struct to store contribution data
    struct ContributionData {
        uint256 contributionScore;
        uint256 roundId;
        uint256 timestamp;
    }
    
    // Mapping to store contributions for each participant
    mapping(address => ContributionData[]) public participantContributions;
    mapping(uint256 => mapping(address => uint256)) public roundContributions;
    address[] public participants;
    mapping(address => bool) public isParticipant;
    
    event ContributionRecorded(address indexed participant, uint256 roundId, uint256 contributionScore);
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }
    
    // Function to record a participant's contribution
    function recordContribution(
        address participant, 
        uint256 roundId, 
        uint256 contributionScore
    ) public onlyOwner {
        // Add participant to list if not already included
        if (!isParticipant[participant]) {
            participants.push(participant);
            isParticipant[participant] = true;
        }
        
        // Store contribution data
        participantContributions[participant].push(
            ContributionData({
                contributionScore: contributionScore,
                roundId: roundId,
                timestamp: block.timestamp
            })
        );
        
        // Update round-specific contribution
        roundContributions[roundId][participant] = contributionScore;
        
        emit ContributionRecorded(participant, roundId, contributionScore);
    }
    
    // Function to get a participant's total contribution across all rounds
    function getTotalContribution(address participant) public view returns (uint256) {
        uint256 total = 0;
        for (uint256 i = 0; i < participantContributions[participant].length; i++) {
            total += participantContributions[participant][i].contributionScore;
        }
        return total;
    }
    
    // Function to get all participants' contributions for a specific round
    function getRoundContributions(uint256 roundId) public view returns (address[] memory, uint256[] memory) {
        uint256[] memory contributions = new uint256[](participants.length);
        
        for (uint256 i = 0; i < participants.length; i++) {
            contributions[i] = roundContributions[roundId][participants[i]];
        }
        
        return (participants, contributions);
    }
    
    // Get all participants
    function getParticipants() public view returns (address[] memory) {
        return participants;
    }
}