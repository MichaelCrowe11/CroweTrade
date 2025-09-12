# CroweTrade AI Trading Infrastructure

This repository contains the comprehensive **Crowe Logic Parallel Financial Agent Ecosystem** - a production-grade cryptocurrency and traditional asset trading platform with advanced AI-driven strategies, risk management, and multi-venue execution capabilities.

## 🚀 Quick Start

### Development Setup
- Install: `pip install -e .[dev]`
- Run tests: `pytest -q`
- Lint/format: `ruff check . && black --check .`

### Crypto Trading Activation
```bash
# Demo mode (safe testing)
python activate_crypto_trading.py --demo

# Live trading (requires API credentials)
export COINBASE_API_KEY="your_api_key"
export COINBASE_API_SECRET="your_secret"
export COINBASE_PASSPHRASE="your_passphrase"
python activate_crypto_trading.py --monitor 60
```

## 🏗️ Architecture Overview

**CroweTrade** implements a sophisticated parallel agent ecosystem designed for quantitative trading across cryptocurrency and traditional markets. The system features:

- **24 AI Research Scientists** (Quant Pods) continuously developing new strategies
- **Multi-Venue Execution** with smart order routing and transaction cost analysis
- **Advanced Risk Management** with real-time VaR, regime detection, and circuit breakers
- **Cryptocurrency Integration** supporting Bitcoin, Ethereum, and major altcoins
- **Cross-Chain Wallet Management** with hot/cold storage and multi-signature security

### Key Components

- `src/crowetrade/` — Core trading infrastructure and live agents
- `src/crowetrade/services/crypto_trading_service.py` — Comprehensive crypto trading orchestrator
- `src/crowetrade/services/execution_service/coinbase_adapter.py` — Coinbase Pro API integration
- `src/crowetrade/data/crypto_market_data.py` — Multi-exchange market data aggregation
- `src/crowetrade/services/crypto_wallet.py` — Multi-network cryptocurrency wallet management
- `src/crowetrade/risk/crypto_risk.py` — Cryptocurrency-specific risk controls
- `config/crypto_trading.yaml` — Comprehensive crypto trading configuration
- `specs/` — JSON schemas and policy examples for core trading events
- `.github/instructions/Architecture.instructions.md` — Complete system blueprint

## 🎯 Crypto Trading Capabilities

### Supported Exchanges
- **Coinbase Pro** - Live trading with WebSocket streaming
- **Binance** - Market data aggregation
- **Kraken** - Market data aggregation

### Supported Assets
- **Bitcoin (BTC)** - Primary cryptocurrency with advanced microstructure analysis
- **Ethereum (ETH)** - Smart contract platform with gas optimization
- **Cardano (ADA)** - Proof-of-stake blockchain
- **Solana (SOL)** - High-performance blockchain
- **Polygon (MATIC)** - Ethereum scaling solution
- **USD Coin (USDC)** - Stablecoin for cash management

### Advanced Features
- **Real-Time Market Data** from multiple exchanges with consensus pricing
- **Cross-Chain Bridge Integration** for Polygon and BSC networks
- **Hardware Wallet Support** with Ledger and Trezor integration
- **Multi-Signature Security** with customizable signing requirements
- **Volatility-Adjusted Position Sizing** using Kelly criterion
- **Regime-Based Risk Management** with market stress detection
- **Advanced Order Types** including VWAP, TWAP, and smart limit ladders

## 📈 Trading Strategies

The system implements multiple quantitative strategies:

1. **Momentum Strategies** - Cross-sectional and time-series momentum with regime awareness
2. **Mean Reversion** - Statistical arbitrage with volatility clustering
3. **Cross-Asset Arbitrage** - Multi-venue price discrepancy exploitation  
4. **Trend Following** - Multi-timeframe trend detection with adaptive filters
5. **Microstructure Alpha** - Order book imbalance and flow toxicity signals

## 🛡️ Risk Management

### Portfolio Risk Controls
- **Value-at-Risk (VaR)** monitoring with 1-day and 7-day horizons
- **Maximum Drawdown** limits with real-time circuit breakers
- **Position Sizing** using Kelly-tempered allocation
- **Correlation Limits** preventing over-concentration
- **Volatility Targeting** with dynamic exposure scaling

### Crypto-Specific Risk Features
- **Regime Detection** identifying high-volatility periods
- **Liquidity Monitoring** across multiple exchanges
- **Cross-Chain Risk** assessment for bridge operations
- **Custody Controls** with hot/cold wallet thresholds

## 🔧 Development Roadmap

### ✅ Completed (Production Ready)
- Core trading infrastructure with parallel agents
- Coinbase Pro integration with real-time WebSocket
- Multi-exchange crypto market data feeds
- Comprehensive crypto wallet management
- Advanced crypto-specific risk controls
- Production deployment infrastructure
- Persistent feature store with policy hot-reload

### 🚧 In Progress
- Additional exchange integrations (Binance, Kraken live trading)
- DeFi protocol integration (Uniswap, Aave)
- Advanced machine learning models
- Real-time performance attribution

### 🎯 Future Enhancements
- Options and derivatives trading
- Cross-chain MEV strategies
- Yield farming automation
- NFT market making

Deployment

- Fly.io (preferred)
	- Export a Fly token as `FLYIO_TOKEN`.
	- Deploy both services with health checks:
		- `tools/deploy.sh fly`

- Local (Docker Compose fallback)
	- Bring up both services locally with health checks:
		- `tools/deploy.sh local`

Health endpoints

- Execution: `https://<host>:<port>/health` (local: `http://localhost:18080/health`)
- Portfolio: `https://<host>:<port>/health` (local: `http://localhost:18081/health`)

## Persistent Feature Store & Policy Hot-Reload

This codebase now includes:

1. Disk-backed Feature Store (`DiskFeatureStore`)
	- Append-only JSONL per key with auto-incrementing version and SHA-256 content hash.
	- Retrieval of historical snapshots: `history(key, limit)` and `get_version(key, n)`.
2. Policy Hot-Reload (`PolicyRegistry`)
	- Automatic mtime detection; policies reloaded when underlying YAML changes.
	- Stable `policy_hash` exposed and attached to every emitted `Signal` for auditability.

Example usage:

```python
from crowetrade.feature_store.store import DiskFeatureStore
from crowetrade.core.policy import PolicyRegistry

fs = DiskFeatureStore("./feature_history")
fs.put("AAPL", {"mom": 0.12, "vol": 0.34})
print(fs.get("AAPL").version)
print(len(fs.history("AAPL")))

registry = PolicyRegistry("specs/policies")
pol = registry.load("cs_mom_v1")
print(registry.policy_hash("cs_mom_v1"))
```

Backward compatibility: `InMemoryFeatureStore.put(key, values, version=...)` still accepted.

