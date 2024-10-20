// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

contract HoshiNFT is ERC721, Ownable {
    uint256 public nextTokenId;

    // Mapping to track parent NFTs for derivative works
    mapping(uint256 => uint256[]) public parentNFTs;

    // Mapping to store the similarity score between child and parent (1-100)
    mapping(uint256 => uint256[]) public similarityScores;

    // Mapping to store token URIs (IPFS links)
    mapping(uint256 => string) private tokenURIs;

    // Mapping from owner address to list of owned token IDs
    mapping(address => uint256[]) private _ownedTokens;

    // Event for new NFT minted
    event NFTMinted(uint256 tokenId, address creator, string tokenURI);

    // Event for linking derivative with similarity score
    event DerivativeLinked(
        uint256 tokenId,
        uint256[] parentIds,
        uint256[] similarityScores
    );

    constructor(
        address initialOwner
    ) ERC721("HoshiNFT", "HSN") Ownable(initialOwner) {
        nextTokenId = 1;
    }

    // helper function to find if tokenId exists
    function checkIfMinted(uint256 tokenId) public view returns (bool) {
        try this.ownerOf(tokenId) returns (address owner) {
            return owner != address(0);
        } catch {
            return false;
        }
    }

    // Mint a new NFT and optionally link it to parent NFTs if it's a derivative
    // Immediate parent and score is at the start of the list
    function mintNFT(
        address creator,
        uint256[] memory parentTokenIds,
        uint256[] memory similarityScoresInput,
        string memory IPFSTokenURI
    ) public returns (uint256) {
        require(
            parentTokenIds.length == similarityScoresInput.length,
            "Parent IDs and similarity scores must match"
        );

        // Check if all parentTokenIds exist before proceeding
        for (uint256 i = 0; i < parentTokenIds.length; i++) {
            require(
                checkIfMinted(parentTokenIds[i]),
                "One or more parent NFTs do not exist"
            );
        }

        uint256 newTokenId = nextTokenId;
        _mint(creator, newTokenId);
        _ownedTokens[creator].push(nextTokenId);
        nextTokenId++;

        // Store the IPFS URI for the newly minted token
        tokenURIs[newTokenId] = IPFSTokenURI;

        emit NFTMinted(newTokenId, creator, IPFSTokenURI);

        // If there are parent NFTs, link them with similarity scores
        if (parentTokenIds.length > 0) {
            for (uint256 i = 0; i < parentTokenIds.length; i++) {
                require(
                    checkIfMinted(parentTokenIds[i]),
                    "Parent NFT does not exist"
                );
                parentNFTs[newTokenId].push(parentTokenIds[i]);
                similarityScores[newTokenId].push(similarityScoresInput[i]);
            }
            emit DerivativeLinked(
                newTokenId,
                parentTokenIds,
                similarityScoresInput
            );
        }

        return newTokenId;
    }

    // Get the token URI for a given token (IPFS link)
    function tokenURI(
        uint256 tokenId
    ) public view override returns (string memory) {
        require(checkIfMinted(tokenId), "URI query for nonexistent token");
        return tokenURIs[tokenId];
    }

    // Get parent NFTs for a given token
    function getParents(
        uint256 tokenId
    ) public view returns (uint256[] memory) {
        return parentNFTs[tokenId];
    }

    // Get the similarity scores for a given token
    function getSimilarityScores(
        uint256 tokenId
    ) public view returns (uint256[] memory) {
        return similarityScores[tokenId];
    }

    // Check if tokenId1 is a derivative of tokenId2
    function isDerivative(
        uint256 tokenId1,
        uint256 tokenId2
    ) public view returns (bool) {
        uint256[] memory parents = parentNFTs[tokenId1];
        for (uint256 i = 0; i < parents.length; i++) {
            if (parents[i] == tokenId2) {
                return true;
            }
        }
        return false;
    }

    function getTokenIdsByOwner(
        address owner
    ) public view returns (uint256[] memory) {
        return _ownedTokens[owner];
    }
}
