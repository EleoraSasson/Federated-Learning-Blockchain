const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying contracts with the account:", deployer.address);

  // Deploy FLRewardToken
  const FLRewardToken = await hre.ethers.getContractFactory("FLRewardToken");
  const flRewardToken = await FLRewardToken.deploy(1000000); // 1 million initial supply
  await flRewardToken.waitForDeployment();
  console.log("FLRewardToken deployed to:", await flRewardToken.getAddress());

  // Get your contribution contract address
  const contributionContractAddress = "0x9fe46736679d2d9a65f0992f2272de9f3c7fa6e0";
  console.log("Using contribution contract at:", contributionContractAddress);

  // Deploy FLRewardDistribution
  const rewardPerRound = hre.ethers.parseEther("1000"); // 1000 tokens per round
  const FLRewardDistribution = await hre.ethers.getContractFactory("FLRewardDistribution");
  const flRewardDistribution = await FLRewardDistribution.deploy(
    await flRewardToken.getAddress(),
    contributionContractAddress,
    rewardPerRound
  );
  await flRewardDistribution.waitForDeployment();
  console.log("FLRewardDistribution deployed to:", await flRewardDistribution.getAddress());

  // Transfer ownership of token contract to distribution contract
  console.log("Transferring token ownership to distribution contract...");
  try {
    const tx = await flRewardToken.transferOwnership(await flRewardDistribution.getAddress());
    await tx.wait();
    console.log("Token ownership transferred to distribution contract");
  } catch (error) {
    console.error("Error transferring ownership:", error.message);
  }

  console.log("Deployment completed");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });