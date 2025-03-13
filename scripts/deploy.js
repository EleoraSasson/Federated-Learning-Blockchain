async function main() {
    // Deploy the communication contract
    const FLComm = await ethers.getContractFactory("FLCommunication");
    const flComm = await FLComm.deploy();
    await flComm.waitForDeployment();
    
    console.log("FLCommunication deployed to:", await flComm.getAddress());
    
    // Deploy the contribution contract
    const FLContrib = await ethers.getContractFactory("FLContribution");
    const flContrib = await FLContrib.deploy();
    await flContrib.waitForDeployment();
    
    console.log("FLContribution deployed to:", await flContrib.getAddress());
    
    // Deploy the reward token (initial supply of 1,000,000 tokens)
    const FLRewardToken = await ethers.getContractFactory("FLRewardToken");
    const flRewardToken = await FLRewardToken.deploy(1000000);
    await flRewardToken.waitForDeployment();
    
    console.log("FLRewardToken deployed to:", await flRewardToken.getAddress());
    
    // Deploy the reward distribution contract
    const FLRewardDistribution = await ethers.getContractFactory("FLRewardDistribution");
    // Set reward per round to 100 tokens
    const flRewardDistribution = await FLRewardDistribution.deploy(
      await flRewardToken.getAddress(),
      await flContrib.getAddress(),
      100
    );
    await flRewardDistribution.waitForDeployment();
    
    console.log("FLRewardDistribution deployed to:", await flRewardDistribution.getAddress());
    
    // Grant the distribution contract permission to mint tokens
    await flRewardToken.transferOwnership(await flRewardDistribution.getAddress());
    console.log("Token ownership transferred to distribution contract");
  }
  
  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });