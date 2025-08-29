"""Main orchestrator for CroweTrade trading system."""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from crowetrade.config.settings import Settings, ConfigManager
from crowetrade.core.bus import EventBus
from crowetrade.data.market_data import MarketDataProcessor, MarketDataAggregator
from crowetrade.data.providers.simulator import SimulatedMarketData, ProviderConfig
from crowetrade.execution.adapters import BrokerExecutionAdapter
from crowetrade.execution.algorithms import TWAPAlgorithm, VWAPAlgorithm
from crowetrade.execution.brokers.interactive_brokers import InteractiveBrokersAdapter
from crowetrade.execution.smart_router import SmartOrderRouter, VenueType
from crowetrade.features.indicators import TechnicalIndicators
from crowetrade.live.portfolio_agent import PortfolioAgent
from crowetrade.live.signal_agent import SignalAgent
from crowetrade.persistence.timescale import TimescaleDB
from crowetrade.risk.integrated_risk import IntegratedRiskSystem
from crowetrade.strategies.momentum import CrossSectionalMomentum
from crowetrade.strategies.mean_reversion import MeanReversionStrategy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CroweTradingSystem:
    """Main orchestrator for the CroweTrade system."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize trading system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config_manager = ConfigManager()
        if config_path:
            self.config_manager.config_path = config_path
            self.config_manager.reload()
        
        self.settings = self.config_manager.settings
        
        # Core components
        self.event_bus = EventBus()
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Initialize components (will be set up in setup())
        self.market_processor: Optional[MarketDataProcessor] = None
        self.market_aggregator: Optional[MarketDataAggregator] = None
        self.database: Optional[TimescaleDB] = None
        self.risk_system: Optional[IntegratedRiskSystem] = None
        self.smart_router: Optional[SmartOrderRouter] = None
        
        # Strategies
        self.momentum_strategy: Optional[CrossSectionalMomentum] = None
        self.mean_reversion_strategy: Optional[MeanReversionStrategy] = None
        
        # Agents
        self.signal_agents: List[SignalAgent] = []
        self.portfolio_agent: Optional[PortfolioAgent] = None
        
        # Market data providers
        self.data_providers = []
        
        # Execution adapters
        self.broker_adapters = {}
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
    
    async def setup(self) -> None:
        """Setup all system components."""
        logger.info("Setting up CroweTrade system...")
        
        # Initialize database
        if self.settings.database:
            self.database = TimescaleDB(self.settings.database)
            await self.database.connect()
        
        # Initialize market data processing
        self.market_processor = MarketDataProcessor()
        self.market_aggregator = MarketDataAggregator()
        
        # Setup market data providers
        await self._setup_data_providers()
        
        # Initialize risk system
        self.risk_system = IntegratedRiskSystem(self.settings.risk)
        
        # Setup execution layer
        await self._setup_execution()
        
        # Initialize strategies
        self._setup_strategies()
        
        # Setup agents
        self._setup_agents()
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info("CroweTrade system setup complete")
    
    async def _setup_data_providers(self) -> None:
        """Setup market data providers."""
        # For now, use simulated data
        sim_config = ProviderConfig(
            name="simulator",
            heartbeat_interval=30,
        )
        
        sim_provider = SimulatedMarketData(
            sim_config,
            self.market_processor,
            tick_rate=10,  # 10 ticks per second
        )
        
        self.data_providers.append(sim_provider)
        
        # Start provider
        await sim_provider.start()
        
        # Subscribe to some symbols
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        await sim_provider.subscribe_trades(symbols)
        await sim_provider.subscribe_quotes(symbols)
        await sim_provider.subscribe_book(symbols)
        await sim_provider.subscribe_bars(symbols, "1m")
    
    async def _setup_execution(self) -> None:
        """Setup execution layer."""
        # Initialize smart router
        self.smart_router = SmartOrderRouter()
        
        # Setup broker adapters
        for broker_config in self.settings.brokers:
            if broker_config.name == "interactive_brokers":
                ib_adapter = InteractiveBrokersAdapter({
                    "host": "localhost",
                    "port": 7497,
                    "client_id": 1,
                    "initial_balance": 100000,
                })
                
                await ib_adapter.connect()
                
                # Create execution adapter
                exec_adapter = BrokerExecutionAdapter(ib_adapter)
                self.broker_adapters[broker_config.name] = exec_adapter
                
                # Register with smart router
                self.smart_router.register_venue(
                    broker_config.name,
                    VenueType.BROKER,
                    exec_adapter,
                )
    
    def _setup_strategies(self) -> None:
        """Setup trading strategies."""
        # Momentum strategy
        self.momentum_strategy = CrossSectionalMomentum(
            lookback_period=20,
            holding_period=5,
            top_percentile=0.2,
            bottom_percentile=0.2,
        )
        
        # Mean reversion strategy
        self.mean_reversion_strategy = MeanReversionStrategy(
            lookback_period=20,
            entry_threshold=2.0,
            exit_threshold=0.5,
        )
    
    def _setup_agents(self) -> None:
        """Setup trading agents."""
        # Portfolio agent
        self.portfolio_agent = PortfolioAgent(
            risk_budget=0.1,
            turnover_penalty=0.001,
        )
        
        # Signal agents for different strategies
        
        # Momentum signal agent
        momentum_model = lambda features: self._momentum_signal_generator(features)
        momentum_agent = SignalAgent(
            model=momentum_model,
            policy_id="momentum_v1",
            gates={"prob_edge_min": 0.55, "sigma_max": 0.03},
        )
        self.signal_agents.append(momentum_agent)
        
        # Mean reversion signal agent
        mean_rev_model = lambda features: self._mean_reversion_signal_generator(features)
        mean_rev_agent = SignalAgent(
            model=mean_rev_model,
            policy_id="mean_reversion_v1",
            gates={"prob_edge_min": 0.60, "sigma_max": 0.02},
        )
        self.signal_agents.append(mean_rev_agent)
    
    def _momentum_signal_generator(self, features: Dict) -> Tuple[float, float, float]:
        """Generate momentum signals from features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tuple of (mu, sigma, prob_edge_positive)
        """
        # Simplified signal generation
        momentum = features.get("momentum", 0)
        rsi = features.get("rsi", 50)
        
        # Calculate expected return
        mu = momentum * 0.1  # Scale momentum
        
        # Estimate uncertainty
        sigma = abs(momentum) * 0.5 + 0.01
        
        # Probability of positive edge
        if momentum > 0 and rsi < 70:
            prob_edge = 0.6 + min(momentum, 0.2)
        elif momentum < 0 and rsi > 30:
            prob_edge = 0.4 - min(abs(momentum), 0.2)
        else:
            prob_edge = 0.5
        
        return mu, sigma, prob_edge
    
    def _mean_reversion_signal_generator(self, features: Dict) -> Tuple[float, float, float]:
        """Generate mean reversion signals from features.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tuple of (mu, sigma, prob_edge_positive)
        """
        z_score = features.get("z_score", 0)
        rsi = features.get("rsi", 50)
        
        # Mean reversion logic
        if abs(z_score) > 2:
            # Strong deviation, expect reversion
            mu = -z_score * 0.01  # Negative correlation
            sigma = abs(z_score) * 0.002 + 0.005
            
            if z_score < -2 and rsi < 30:
                prob_edge = 0.7  # Oversold
            elif z_score > 2 and rsi > 70:
                prob_edge = 0.7  # Overbought
            else:
                prob_edge = 0.6
        else:
            mu = 0
            sigma = 0.01
            prob_edge = 0.5
        
        return mu, sigma, prob_edge
    
    def _register_event_handlers(self) -> None:
        """Register event handlers for system events."""
        
        async def on_market_trade(trade):
            """Handle market trade events."""
            # Store in database if available
            if self.database:
                await self.database.insert_trade(trade)
        
        async def on_market_quote(quote):
            """Handle market quote events."""
            if self.database:
                await self.database.insert_quote(quote)
        
        async def on_market_bar(bar):
            """Handle market bar events."""
            if self.database:
                await self.database.insert_bar(bar)
            
            # Run strategies on new bars
            await self._run_strategies(bar)
        
        # Subscribe to events
        asyncio.create_task(
            self.event_bus.subscribe("market.trade", on_market_trade)
        )
        asyncio.create_task(
            self.event_bus.subscribe("market.quote", on_market_quote)
        )
        asyncio.create_task(
            self.event_bus.subscribe("market.bar", on_market_bar)
        )
    
    async def _run_strategies(self, bar) -> None:
        """Run trading strategies on new market data.
        
        Args:
            bar: New bar data
        """
        # This is simplified - in production would aggregate across symbols
        # and run strategies periodically
        pass
    
    async def run(self) -> None:
        """Run the trading system."""
        logger.info("Starting CroweTrade trading system...")
        
        self.is_running = True
        
        try:
            # Setup system
            await self.setup()
            
            # Main event loop
            while self.is_running:
                # Check for shutdown
                if self.shutdown_event.is_set():
                    break
                
                # Process events
                await asyncio.sleep(0.1)
                
                # Update risk metrics
                if self.risk_system:
                    metrics = self.risk_system.get_risk_metrics()
                    
                    # Log key metrics periodically
                    if datetime.utcnow().second == 0:  # Once per minute
                        logger.info(
                            f"Risk Metrics - "
                            f"Drawdown: {metrics['drawdown']:.2%}, "
                            f"Daily PnL: {metrics['daily_pnl']:.2f}, "
                            f"Positions: {metrics['positions']}"
                        )
        
        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
        
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Shutdown the trading system."""
        logger.info("Shutting down CroweTrade system...")
        
        self.is_running = False
        
        # Stop data providers
        for provider in self.data_providers:
            await provider.stop()
        
        # Disconnect brokers
        for adapter_name, adapter in self.broker_adapters.items():
            if hasattr(adapter.broker, 'disconnect'):
                await adapter.broker.disconnect()
        
        # Disconnect database
        if self.database:
            await self.database.disconnect()
        
        logger.info("CroweTrade shutdown complete")


async def main():
    """Main entry point."""
    # Parse command line arguments if needed
    config_path = Path("config/development.yaml")
    
    # Create and run system
    system = CroweTradingSystem(config_path)
    await system.run()


if __name__ == "__main__":
    asyncio.run(main())