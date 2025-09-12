// Coinbase Developer Platform Integration for CroweTrade
// Provides onchain capabilities and Smart Wallet integration

import { OnchainKitProvider } from '@coinbase/onchainkit';
import { base } from 'wagmi/chains';

// Coinbase Developer Platform Configuration
export const COINBASE_CONFIG = {
  projectId: process.env.NEXT_PUBLIC_CDP_PROJECT_ID || 'b1e3b0af-35cb-48f0-aec7-276d3c4fbf79',
  chain: base,
  rpcUrl: process.env.NEXT_PUBLIC_CDP_RPC_URL || `https://api.developer.coinbase.com/rpc/v1/base/${process.env.NEXT_PUBLIC_CDP_PROJECT_ID}`,
};

// OnchainKit Provider Component
export function CoinbaseProvider({ children }: { children: React.ReactNode }) {
  return (
    <OnchainKitProvider
      apiKey={process.env.NEXT_PUBLIC_CDP_API_KEY}
      chain={COINBASE_CONFIG.chain}
      config={{
        appearance: {
          mode: 'auto',
          theme: 'default',
        },
        wallet: {
          display: 'modal',
        },
      }}
    >
      {children}
    </OnchainKitProvider>
  );
}

// Smart Wallet Integration
export const SmartWalletConfig = {
  // Enable Smart Wallet features
  capabilities: {
    paymasterService: {
      url: `https://api.developer.coinbase.com/rpc/v1/base/${COINBASE_CONFIG.projectId}`,
    },
    bundlerService: {
      url: `https://api.developer.coinbase.com/rpc/v1/base/${COINBASE_CONFIG.projectId}`,
    },
  },
};

// Onchain Trading Utilities
export class OnchainTradingService {
  private projectId: string;
  private apiKey: string;

  constructor() {
    this.projectId = process.env.NEXT_PUBLIC_CDP_PROJECT_ID || COINBASE_CONFIG.projectId;
    this.apiKey = process.env.NEXT_PUBLIC_CDP_API_KEY || '';
  }

  // Execute onchain trades via Coinbase Developer Platform
  async executeOnchainTrade(params: {
    fromToken: string;
    toToken: string;
    amount: string;
    slippage: number;
  }) {
    try {
      // Integration with CroweTrade's existing trading engine
      const response = await fetch('/api/trading/onchain', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CDP-Project-Id': this.projectId,
        },
        body: JSON.stringify(params),
      });

      return await response.json();
    } catch (error) {
      console.error('Onchain trading error:', error);
      throw error;
    }
  }

  // Get onchain portfolio data
  async getOnchainPortfolio(address: string) {
    try {
      const response = await fetch(`/api/portfolio/onchain/${address}`, {
        headers: {
          'X-CDP-Project-Id': this.projectId,
        },
      });

      return await response.json();
    } catch (error) {
      console.error('Onchain portfolio error:', error);
      throw error;
    }
  }

  // Smart Wallet transaction building
  async buildSmartWalletTransaction(params: {
    to: string;
    value: string;
    data: string;
  }) {
    // Integrate with Smart Wallet capabilities
    return {
      ...params,
      gasless: true, // Use paymaster for gasless transactions
      bundled: true, // Bundle with other operations
    };
  }
}

// Coinbase Advanced Trade API Integration (Pro features)
export class CoinbaseAdvancedTradeService {
  private baseUrl = 'https://api.coinbase.com/api/v3/brokerage';
  
  constructor(
    private apiKey: string,
    private apiSecret: string,
    private passphrase: string
  ) {}

  // Get real-time prices for onchain correlation
  async getAdvancedTradePrices(productIds: string[]) {
    // This integrates with our existing Coinbase Pro adapter
    // but adds the new Advanced Trade API capabilities
    const endpoint = '/market/products';
    const response = await this.authenticatedRequest('GET', endpoint);
    return response;
  }

  private async authenticatedRequest(method: string, endpoint: string, body?: any) {
    // HMAC authentication for Coinbase Advanced Trade API
    const timestamp = Math.floor(Date.now() / 1000);
    const message = `${timestamp}${method}${endpoint}${body || ''}`;
    
    // Use the existing HMAC implementation from our trading service
    const signature = this.generateSignature(message, this.apiSecret);
    
    return fetch(`${this.baseUrl}${endpoint}`, {
      method,
      headers: {
        'CB-ACCESS-KEY': this.apiKey,
        'CB-ACCESS-SIGN': signature,
        'CB-ACCESS-TIMESTAMP': timestamp.toString(),
        'CB-ACCESS-PASSPHRASE': this.passphrase,
        'Content-Type': 'application/json',
      },
      body: body ? JSON.stringify(body) : undefined,
    });
  }

  private generateSignature(message: string, secret: string): string {
    // Reuse existing HMAC-SHA256 implementation
    const crypto = require('crypto');
    return crypto.createHmac('sha256', Buffer.from(secret, 'base64'))
      .update(message)
      .digest('base64');
  }
}

// Export all services
export {
  OnchainTradingService,
  CoinbaseAdvancedTradeService,
};

// Environment variables needed for Coinbase Developer Platform
/*
Add to your .env.local:

NEXT_PUBLIC_CDP_PROJECT_ID=b1e3b0af-35cb-48f0-aec7-276d3c4fbf79
NEXT_PUBLIC_CDP_API_KEY=your_coinbase_developer_platform_api_key
NEXT_PUBLIC_CDP_RPC_URL=https://api.developer.coinbase.com/rpc/v1/base/b1e3b0af-35cb-48f0-aec7-276d3c4fbf79

# For server-side Advanced Trade API
COINBASE_ADVANCED_API_KEY=your_advanced_trade_api_key
COINBASE_ADVANCED_SECRET=your_advanced_trade_secret
COINBASE_ADVANCED_PASSPHRASE=your_advanced_trade_passphrase
*/
