// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "./HoshiNFT.sol";

contract HoshiToken is ERC20, ERC20Burnable, Ownable {
    HoshiNFT public HoshiNFTContract;

    address public liquidityPool;

    uint256 public constant TOKENS_PER_ETH = 1000000000; // 1 billion tokens per 1 ETH

    uint256 public subscriptionFeeETH = 1000000 wei; // Fee to subscribe (1 million wei = 19.71 usd)

    mapping(address => bool) public subscribedCreators;

    constructor(
        address initialOwner,
        address _HoshiNFTAddress
    ) ERC20("HoshiToken", "HST") Ownable(initialOwner) {
        liquidityPool = address(this); // Liquidity pool is this contract itself
        HoshiNFTContract = HoshiNFT(_HoshiNFTAddress);
    }

    // Mint tokens for the platform, backed by ETH
    function mintTokens(address to, uint256 amount) public onlyOwner {
        _mint(to, amount);
    }

    // Deposit ETH into the contract and mint tokens into the liquidity pool
    function depositETH() external payable {
        uint256 mintAmount = msg.value * TOKENS_PER_ETH; // Mint based on ETH deposited
        _mint(liquidityPool, mintAmount);
    }

    function distributeTokensEqually(
        address[] memory recipients,
        uint256 totalAmount
    ) public onlyOwner {
        require(recipients.length > 0, "No recipients specified");
        require(totalAmount > 0, "Total amount must be greater than zero");

        // Calculate the equal share for each recipient using decimals
        uint256 shareAmount = totalAmount / recipients.length;

        // Distribute tokens to each recipient equally
        for (uint256 i = 0; i < recipients.length; i++) {
            _transfer(liquidityPool, recipients[i], shareAmount);
        }

        // Handling the remainder by giving the leftover tokens to the last recipient
        uint256 remainder = totalAmount % recipients.length;
        if (remainder > 0) {
            _transfer(
                liquidityPool,
                recipients[recipients.length - 1],
                remainder
            );
        }
    }

    // Distribute royalties based on the similarity scores up the chain of derivatives after a user likes a post
    function likePost(uint256 tokenId, uint256 totalAmount) public {
        uint256 currentTokenId = tokenId;
        uint256 remainingAmount = totalAmount; // Start with the full amount of tokens

        require(balanceOf(msg.sender) >= totalAmount, "Not enough tokens");

        // Fetch the full ancestry list (parents and similarity scores)
        uint256[] memory parentIds = HoshiNFTContract.getParents(
            currentTokenId
        );
        uint256[] memory similarityScores = HoshiNFTContract
            .getSimilarityScores(currentTokenId);

        // Ensure there's a matching similarity score for each parent
        require(
            parentIds.length == similarityScores.length,
            "Mismatch between parent count and similarity scores of token id"
        );

        // Transfer tokens from the user to the contract for distribution
        _transfer(msg.sender, address(this), totalAmount);

        // Loop over parents and distribute royalties
        for (uint256 i = 0; i < parentIds.length; i++) {
            address currentOwner = HoshiNFTContract.ownerOf(currentTokenId);

            // Get the current similarity score (immediate parent)
            uint256 similarityScore = similarityScores[i];

            // Calculate the percentage the current owner will keep (100 - similarityScore)
            uint256 keepPercentage = 100 - similarityScore;
            uint256 keepAmount = (remainingAmount * keepPercentage) / 100;

            // Transfer the amount that the current owner will keep
            _transfer(address(this), currentOwner, keepAmount);

            // Update the remaining amount to pass up to the parent
            remainingAmount -= keepAmount;

            // Move to the next parent in the ancestry
            currentTokenId = parentIds[i];
        }

        // The last parent (original creator) gets the remaining tokens
        address finalOwner = HoshiNFTContract.ownerOf(currentTokenId);
        _transfer(address(this), finalOwner, remainingAmount); // Send remaining tokens to original creator
    }

    // Unwrap tokens to claim ETH (only for subscribed creators)
    function unwrapTokens(uint256 tokenAmount) public {
        require(subscribedCreators[msg.sender], "Not subscribed");
        uint256 ethAmount = tokenAmount / TOKENS_PER_ETH; // Convert tokens to ETH

        // Ensure the contract has enough ETH to fulfill the request
        require(
            address(this).balance >= ethAmount,
            "Insufficient ETH in contract"
        );

        // Burn the tokens
        _burn(msg.sender, tokenAmount);

        // Transfer the corresponding ETH to the creator
        payable(msg.sender).transfer(ethAmount);
    }

    // Subscribe a creator for unwrapping tokens (requires a subscription fee)
    function subscribeCreator() public payable {
        require(msg.value == subscriptionFeeETH, "Incorrect subscription fee");

        // Mark the creator as subscribed
        subscribedCreators[msg.sender] = true;

        // Add the subscription fee to the liquidity pool
        payable(liquidityPool).transfer(msg.value);
    }

    // Update the subscription fee (onlyOwner can call this)
    function setSubscriptionFee(uint256 newFee) external onlyOwner {
        subscriptionFeeETH = newFee;
    }

    // Transfer tokens for interactions (likes, tips)
    function transfer(
        address recipient,
        uint256 amount
    ) public override returns (bool) {
        _transfer(_msgSender(), recipient, amount);
        return true;
    }

    // Transfer tokens and reimburse gas costs to relayer
    function transferAndCoverGas(
        address recipient,
        uint256 amount,
        uint256 gasFee
    ) external onlyOwner {
        _transfer(liquidityPool, recipient, amount); // Transfer tokens to the recipient (user or creator)
        _transfer(liquidityPool, tx.origin, gasFee); // Reimburse the relayer for gas fees
    }

    // Function to check the ETH balance of the contract
    function getContractETHBalance() public view returns (uint256) {
        return address(this).balance;
    }

    // Allow the contract to receive ETH deposits
    receive() external payable {}
}
