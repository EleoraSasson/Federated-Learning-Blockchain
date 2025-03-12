const { expect } = require("chai");

describe("Federated Learning Contracts", function () {
  let flComm;
  let flContrib;
  let owner;
  let client1;
  let client2;
  let client3;

  beforeEach(async function () {
    // Get signers (accounts)
    [owner, client1, client2, client3] = await ethers.getSigners();
    
    // Deploy the communication contract
    const FLComm = await ethers.getContractFactory("FLCommunication");
    flComm = await FLComm.deploy();
    
    // Deploy the contribution contract
    const FLContrib = await ethers.getContractFactory("FLContribution");
    flContrib = await FLContrib.deploy();
  });

  it("Should allow initiating a round", async function () {
    const roundId = 1;
    const publicKey = ethers.encodeBytes32String("test-public-key");
    
    // Initiate a round
    await flComm.initiateRound(roundId, publicKey);
    
    // Add assertions as needed
    expect(await flComm.owner()).to.equal(owner.address);
  });

  it("Should record contributions correctly", async function () {
    const roundId = 1;
    const contributionScore = 100;
    
    await flContrib.recordContribution(client1.address, roundId, contributionScore);
    
    const totalContribution = await flContrib.getTotalContribution(client1.address);
    expect(totalContribution).to.equal(contributionScore);
  });
});