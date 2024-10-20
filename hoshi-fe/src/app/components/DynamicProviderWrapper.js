'use client';

import {
  DynamicContextProvider,
  mergeNetworks,
} from '@dynamic-labs/sdk-react-core';
import { EthereumWalletConnectors } from '@dynamic-labs/ethereum';

export default function DynamicProviderWrapper({ children }) {
  const myEvmNetworks = [
    {
      blockExplorerUrls: [
        'https://giant-half-dual-testnet.explorer.testnet.skalenodes.com',
      ],
      chainId: 974399131,
      chainName: 'SKALE Calypso Hub Testnet',
      iconUrls: ['https://app.dynamic.xyz/assets/networks/eth.svg'],
      name: 'SKALE Calypso Hub Testnet',
      nativeCurrency: {
        decimals: 18,
        name: 'sFUEL',
        symbol: 'sFUEL',
        iconUrl: 'https://app.dynamic.xyz/assets/networks/eth.svg',
      },
      networkId: 974399131,
      rpcUrls: ['https://testnet.skalenodes.com/v1/giant-half-dual-testnet'],
    },
  ];

  return (
    <DynamicContextProvider
      settings={{
        environmentId: process.env.NEXT_PUBLIC_DYNAMIC_ENV_ID,
        walletConnectors: [EthereumWalletConnectors],
        overrides: {
          evmNetworks: (networks) => mergeNetworks(myEvmNetworks, networks),
        },
      }}
    >
      {children}
    </DynamicContextProvider>
  );
}
