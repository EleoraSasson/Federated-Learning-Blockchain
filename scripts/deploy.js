async function main() {
    // Deploy the communication contract
    const FLComm = await ethers.getContractFactory("FLCommunication");
    const flComm = await FLComm.deploy();
    await flComm.waitForDeployment();
    
    console.log("FederatedLearningCommunication deployed to:", await flComm.getAddress());
    
    // Deploy the contribution contract
    const FLContrib = await ethers.getContractFactory("FLContribution");
    const flContrib = await FLContrib.deploy();
    await flContrib.waitForDeployment();
    
    console.log("FederatedLearningContribution deployed to:", await flContrib.getAddress());
  }
  
  main()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });