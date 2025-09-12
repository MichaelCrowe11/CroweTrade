# CroweLang Integration for CroweTrade

This directory contains the CroweLang integration components for the CroweTrade platform.

## Overview

CroweLang is now the primary computing language for CroweTrade, providing a domain-specific language (DSL) for quantitative trading with seamless integration into CroweTrade's Python-based parallel financial agent ecosystem.

## Components

### 1. Language Specification
- `CROWETRADE_INTEGRATION.md` - Complete integration specification and documentation

### 2. Code Generation
- `codegen.python.ts` - Python code generator for CroweTrade agents

### 3. Example Strategies
- `crowetrade-agent.crowe` - Example multi-agent trading strategy

## Features

- **Multi-target compilation**: Python, TypeScript, C++, Rust
- **Agent-based architecture**: Parallel execution with event-driven communication
- **Financial primitives**: Comprehensive market data and order types
- **Risk management**: Advanced position and portfolio risk controls
- **Execution algorithms**: TWAP, VWAP, POV, Iceberg implementations
- **Backtesting framework**: Realistic simulation with CroweTrade data

## Installation

```bash
# Install CroweLang compiler
npm install -g crowelang

# Compile a strategy for CroweTrade
crowelang compile strategy.crowe --target python --crowetrade
```

## Usage Example

```crowelang
// Define a trading agent
agent TradingAgent {
  contract {
    name = "momentum_trader";
    version = "1.0.0";
    requires = ["market_data", "execution"];
  }
  
  indicators {
    rsi = RSI(close, 14);
    macd = MACD(close, 12, 26, 9);
  }
  
  signals {
    long_signal = rsi < 30 and macd.histogram > 0;
  }
  
  rules {
    when (long_signal) {
      execute_trade("buy", calculate_size());
    }
  }
}
```

## Python Integration

The generated Python code integrates seamlessly with CroweTrade's infrastructure:

```python
from crowetrade.contracts import BaseAgent
from crowetrade.execution import ExecutionEngine
from crowetrade.risk import RiskManager

class TradingAgent(BaseAgent):
    """Generated from CroweLang"""
    
    async def on_bar(self, bar: Bar):
        # Process market data
        self.update_indicators(bar)
        
        # Evaluate signals
        if self.evaluate_long_signal():
            # Execute trade with risk checks
            await self.execute_trade("buy", self.calculate_size())
```

## Documentation

For complete documentation, see:
- [CroweLang Documentation](https://crowelang.com/docs)
- [CroweTrade Developer Portal](https://crowetrade.com/developers)
- [Integration Guide](./CROWETRADE_INTEGRATION.md)

## Support

- GitHub: https://github.com/MichaelCrowe11/crowe-lang
- CroweTrade Support: support@crowetrade.com
- Discord: https://discord.gg/crowelang

## License

This integration is proprietary to CroweTrade. The CroweLang compiler is open source under the MIT License.