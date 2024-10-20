// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

// Simple ERC-20 Token representing ETH on the SKALE network
contract SkaleETH is ERC20, Ownable {
    // Constructor to initialize the token with a name and symbol
    constructor(
        address initialOwner
    ) ERC20("SKALE ETH", "sETH") Ownable(initialOwner) {
        // Optionally, you can mint some initial supply to the deployer or another address
        _mint(msg.sender, 10 * (10 ** 18));
    }

    // Function to mint more tokens (only the owner can mint)
    function mint(address to, uint256 amount) external onlyOwner {
        _mint(to, amount);
    }

    // Function to burn tokens (only the owner can burn)
    function burn(uint256 amount) external onlyOwner {
        _burn(msg.sender, amount);
    }
}
