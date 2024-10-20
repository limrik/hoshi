import { http } from 'viem';
import { Account, privateKeyToAccount, Address } from 'viem/accounts';
import { StoryClient, StoryConfig } from "@story-protocol/core-sdk";
import { toHex } from 'viem';

const privateKey = `0x${process.env.WALLET_PRIVATE_KEY}`;
const account = privateKeyToAccount(privateKey);

const config = {
  account: account, // the account object from above
  transport: http(process.env.RPC_PROVIDER_URL),
  chainId: 'iliad'
};

const client = StoryClient.newClient(config);



const response = await client.ipAsset.register({
  nftContract: "0x0FDd174d8A809F93E26804A29080A678b874f064", // your NFT contract address
  tokenId: "12", // your NFT token ID
  ipMetadata: {
    ipMetadataURI: 'test-uri',
    ipMetadataHash: toHex('test-metadata-hash', { size: 32 }),
    nftMetadataHash: toHex('test-nft-metadata-hash', { size: 32 }),
    nftMetadataURI: 'test-nft-uri',
  },
  txOptions: { waitForTransaction: true }
});

console.log(`Root IPA created at transaction hash ${response.txHash}, IPA ID: ${response.ipId}`)

module.exports = { client };