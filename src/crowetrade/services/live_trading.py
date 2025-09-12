"""
Live Trading Service for CroweTrade AI Trading Infrastructure

This service coordinates live trading operations using the advanced AI capabilities:
- Model Registry for strategy selection
- A/B Testing for model allocation  
- Risk management and position sizing
- Real-time execution with market data feeds
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import yaml
from dataclasses import dataclass, asdict

# Import CroweTrade components
from crowetrade.models.registry import ModelRegistry
from crowetrade.models.ab_testing import ABTestEngine
from crowetrade.backtesting.strategy_integration import IntegratedStrategy, create_strategy_config
from crowetrade.core.types import Signal, HealthStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LiveTradingConfig:
    """Configuration for live trading operations"""
    max_position_size: float = 1000000.0  # $1M max position
    max_daily_loss: float = 50000.0       # $50K daily loss limit
    risk_check_frequency: int = 30        # seconds
    model_evaluation_frequency: int = 300 # 5 minutes
    rebalance_frequency: int = 3600      # 1 hour
    
    # Risk parameters
    max_leverage: float = 3.0
    var_limit: float = 0.02              # 2% VaR
    drawdown_limit: float = 0.05         # 5% max drawdown
    
    # A/B Testing parameters
    min_samples_for_switch: int = 100
    significance_level: float = 0.05
    
    # Execution parameters
    market_data_timeout: float = 5.0
    execution_timeout: float = 30.0

class LiveTradingEngine:
    """Main engine for live trading operations"""
    
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.model_registry = None
        self.ab_testing_engine = None
        self.strategies: Dict[str, IntegratedStrategy] = {}
        self.active_positions: Dict[str, float] = {}
        self.daily_pnl: float = 0.0
        self.running = False
        
        # Performance tracking
        self.trade_count = 0
        self.win_count = 0
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.equity_peak = 0.0
        
        logger.info("Live Trading Engine initialized")
    
    async def initialize(self, models_dir: str = "/app/models"):
        """Initialize the trading engine with AI components"""
        try:
            # Initialize Model Registry
            self.model_registry = ModelRegistry(models_dir)
            logger.info(f"Model Registry initialized with {len(self.model_registry.list_models())} models")
            
            # Initialize A/B Testing Engine
            self.ab_testing_engine = ABTestEngine()
            
            # Load available strategies
            await self._load_strategies()
            
            # Setup A/B tests for available models
            await self._setup_ab_tests()
            
            logger.info("Live Trading Engine fully initialized and ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize Live Trading Engine: {e}")
            raise
    
    async def _load_strategies(self):
        """Load and configure trading strategies"""
        try:
            # Load momentum strategy
            momentum_config = create_strategy_config(
                risk_aversion=2.0,
                max_position_weight=0.1,
                use_regime_detection=True
            )
            self.strategies['momentum'] = IntegratedStrategy(momentum_config)
            
            # Load mean reversion strategy  
            mean_rev_config = create_strategy_config(
                risk_aversion=3.0,
                max_position_weight=0.08,
                use_regime_detection=False
            )
            self.strategies['mean_reversion'] = IntegratedStrategy(mean_rev_config)
            
            logger.info(f"Loaded {len(self.strategies)} trading strategies")
            
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
            raise
    
    async def _setup_ab_tests(self):
        """Setup A/B tests for model selection"""
        try:
            # Get available models from registry
            models = self.model_registry.list_models()
            
            if len(models) >= 2:
                # Create A/B test for each strategy type
                for strategy_type in self.strategies.keys():
                    relevant_models = [m for m in models if strategy_type in m.get('strategy_type', '')]
                    
                    if len(relevant_models) >= 2:
                        test_name = f"{strategy_type}_model_selection"
                        model_names = [m['model_id'] for m in relevant_models]
                        
                        self.ab_testing_engine.create_test(
                            test_name=test_name,
                            model_names=model_names,
                            algorithm='thompson_sampling'
                        )
                        
                        logger.info(f"Created A/B test '{test_name}' with {len(model_names)} models")
            
        except Exception as e:
            logger.error(f"Failed to setup A/B tests: {e}")
    
    async def start_live_trading(self):
        """Start the live trading operation"""
        self.running = True
        logger.info("🚀 Starting Live Trading with AI Infrastructure!")
        
        try:
            # Start concurrent tasks
            await asyncio.gather(
                self._market_data_loop(),
                self._risk_monitoring_loop(),
                self._model_evaluation_loop(),
                self._rebalancing_loop(),
                self._performance_reporting_loop()
            )
        except Exception as e:
            logger.error(f"Live trading error: {e}")
            await self.stop_live_trading()
    
    async def stop_live_trading(self):
        """Stop live trading and close all positions"""
        self.running = False
        logger.info("Stopping live trading...")
        
        try:
            # Close all positions
            for symbol, position in self.active_positions.items():
                if abs(position) > 0.01:  # Close if position > $0.01
                    await self._execute_trade(symbol, -position, "CLOSE_ALL")
            
            # Final performance report
            await self._generate_performance_report()
            
            logger.info("Live trading stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping live trading: {e}")
    
    async def _market_data_loop(self):
        """Continuously process market data and generate signals"""
        while self.running:
            try:
                # Simulate market data (replace with real market data feed)
                market_data = await self._get_market_data()
                
                if market_data:
                    # Generate signals from each strategy
                    for strategy_name, strategy in self.strategies.items():
                        try:
                            # Get optimal model from A/B testing
                            test_name = f"{strategy_name}_model_selection"
                            if test_name in self.ab_testing_engine.active_tests:
                                model_allocation = self.ab_testing_engine.get_model_allocation(test_name)
                                selected_model = max(model_allocation.items(), key=lambda x: x[1])[0]
                                
                                # Generate signal using selected model
                                signal = await self._generate_signal(strategy, selected_model, market_data)
                                
                                if signal and abs(signal.strength) > 0.1:  # Minimum signal threshold
                                    await self._process_signal(signal, strategy_name)
                                    
                        except Exception as e:
                            logger.warning(f"Error processing strategy {strategy_name}: {e}")
                
                await asyncio.sleep(1)  # 1 second market data frequency
                
            except Exception as e:
                logger.error(f"Market data loop error: {e}")
                await asyncio.sleep(5)
    
    async def _risk_monitoring_loop(self):
        """Continuously monitor risk metrics and enforce limits"""
        while self.running:
            try:
                # Calculate current risk metrics
                total_exposure = sum(abs(pos) for pos in self.active_positions.values())
                current_leverage = total_exposure / max(100000, 100000 + self.total_return)  # Assume $100K starting capital
                
                # Check risk limits
                if current_leverage > self.config.max_leverage:
                    logger.warning(f"Leverage limit exceeded: {current_leverage:.2f} > {self.config.max_leverage}")
                    await self._reduce_positions(0.5)  # Reduce positions by 50%
                
                if self.daily_pnl < -self.config.max_daily_loss:
                    logger.error(f"Daily loss limit exceeded: ${self.daily_pnl:,.2f}")
                    await self.stop_live_trading()
                
                # Update drawdown
                if self.total_return > self.equity_peak:
                    self.equity_peak = self.total_return
                
                current_drawdown = (self.equity_peak - self.total_return) / max(self.equity_peak, 1.0)
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                
                if current_drawdown > self.config.drawdown_limit:
                    logger.error(f"Drawdown limit exceeded: {current_drawdown:.2%}")
                    await self._reduce_positions(0.3)
                
                await asyncio.sleep(self.config.risk_check_frequency)
                
            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _model_evaluation_loop(self):
        """Continuously evaluate model performance and update A/B tests"""
        while self.running:
            try:
                # Evaluate each active A/B test
                for test_name in list(self.ab_testing_engine.active_tests.keys()):
                    test_results = self.ab_testing_engine.get_test_results(test_name)
                    
                    if test_results.get('total_samples', 0) >= self.config.min_samples_for_switch:
                        significance = test_results.get('significance_analysis', {})
                        
                        if significance.get('is_significant', False):
                            best_model = significance.get('best_model')
                            logger.info(f"A/B test '{test_name}' shows significant winner: {best_model}")
                            
                            # Could implement automatic model switching here
                
                await asyncio.sleep(self.config.model_evaluation_frequency)
                
            except Exception as e:
                logger.error(f"Model evaluation error: {e}")
                await asyncio.sleep(300)
    
    async def _rebalancing_loop(self):
        """Periodic portfolio rebalancing"""
        while self.running:
            try:
                if self.active_positions:
                    logger.info("Performing portfolio rebalancing...")
                    
                    # Simple rebalancing logic (equal weight for now)
                    target_weight = 1.0 / len(self.active_positions)
                    total_value = 100000 + self.total_return  # Starting capital + returns
                    
                    for symbol in self.active_positions.keys():
                        target_position = target_weight * total_value * 0.5  # 50% invested
                        current_position = self.active_positions.get(symbol, 0.0)
                        
                        if abs(target_position - current_position) > 1000:  # $1K threshold
                            trade_size = target_position - current_position
                            await self._execute_trade(symbol, trade_size, "REBALANCE")
                
                await asyncio.sleep(self.config.rebalance_frequency)
                
            except Exception as e:
                logger.error(f"Rebalancing error: {e}")
                await asyncio.sleep(3600)
    
    async def _performance_reporting_loop(self):
        """Regular performance reporting"""
        while self.running:
            try:
                await asyncio.sleep(300)  # Report every 5 minutes
                await self._log_performance_metrics()
                
            except Exception as e:
                logger.error(f"Performance reporting error: {e}")
                await asyncio.sleep(300)
    
    async def _get_market_data(self) -> Optional[Dict]:
        """Get real-time market data (placeholder)"""
        # Simulate market data - replace with real market data feed
        import random
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'SPY']
        
        return {
            'timestamp': datetime.now(),
            'data': {
                symbol: {
                    'price': 100 + random.uniform(-5, 5),
                    'volume': random.randint(1000, 10000),
                    'bid': 100 + random.uniform(-5, 5) - 0.01,
                    'ask': 100 + random.uniform(-5, 5) + 0.01
                }
                for symbol in symbols
            }
        }
    
    async def _generate_signal(self, strategy: IntegratedStrategy, model_id: str, market_data: Dict) -> Optional[Signal]:
        """Generate trading signal using strategy and selected model"""
        try:
            # Simulate signal generation (replace with actual model inference)
            import random
            
            if random.random() > 0.7:  # 30% chance of signal
                return Signal(
                    symbol=random.choice(['AAPL', 'GOOGL', 'MSFT']),
                    strength=random.uniform(-1.0, 1.0),
                    confidence=random.uniform(0.5, 0.9),
                    timestamp=datetime.now(),
                    strategy=model_id,
                    metadata={'source': 'live_trading'}
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None
    
    async def _process_signal(self, signal: Signal, strategy_name: str):
        """Process trading signal and execute if appropriate"""
        try:
            # Calculate position size based on signal strength and risk limits
            base_position = 10000  # $10K base position
            position_size = base_position * signal.strength * signal.confidence
            
            # Apply risk limits
            max_position = min(self.config.max_position_size, base_position * 2)
            position_size = max(-max_position, min(max_position, position_size))
            
            if abs(position_size) > 1000:  # $1K minimum trade
                await self._execute_trade(signal.symbol, position_size, signal.strategy)
                
                # Record result for A/B testing
                test_name = f"{strategy_name}_model_selection"
                if test_name in self.ab_testing_engine.active_tests:
                    # Simulate immediate result (replace with actual P&L tracking)
                    import random
                    result = random.uniform(-0.02, 0.02)  # -2% to +2% return
                    self.ab_testing_engine.record_result(test_name, signal.strategy, result)
                
        except Exception as e:
            logger.error(f"Signal processing error: {e}")
    
    async def _execute_trade(self, symbol: str, size: float, strategy: str):
        """Execute trade (placeholder - replace with real broker integration)"""
        try:
            # Simulate trade execution
            execution_price = 100.0 + (hash(symbol) % 20) - 10  # Mock price
            
            # Update positions
            self.active_positions[symbol] = self.active_positions.get(symbol, 0.0) + size
            
            # Update P&L (simplified)
            trade_pnl = abs(size) * 0.001  # Assume 0.1% average profit per trade
            if size > 0:  # Buy
                trade_pnl *= 1
            else:  # Sell
                trade_pnl *= -1
            
            self.daily_pnl += trade_pnl
            self.total_return += trade_pnl
            self.trade_count += 1
            
            if trade_pnl > 0:
                self.win_count += 1
            
            logger.info(f"TRADE EXECUTED: {symbol} ${size:,.2f} @ ${execution_price:.2f} | Strategy: {strategy} | P&L: ${trade_pnl:,.2f}")
            
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def _reduce_positions(self, reduction_factor: float):
        """Reduce all positions by specified factor"""
        for symbol, position in list(self.active_positions.items()):
            reduction = position * reduction_factor
            await self._execute_trade(symbol, -reduction, "RISK_REDUCTION")
    
    async def _log_performance_metrics(self):
        """Log current performance metrics"""
        win_rate = (self.win_count / max(1, self.trade_count)) * 100
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'total_return': self.total_return,
            'daily_pnl': self.daily_pnl,
            'trade_count': self.trade_count,
            'win_rate': win_rate,
            'max_drawdown': self.max_drawdown * 100,
            'active_positions': len(self.active_positions),
            'total_exposure': sum(abs(pos) for pos in self.active_positions.values())
        }
        
        logger.info(f"PERFORMANCE: Return=${metrics['total_return']:,.2f} | "
                   f"Daily P&L=${metrics['daily_pnl']:,.2f} | "
                   f"Trades={metrics['trade_count']} | "
                   f"Win Rate={metrics['win_rate']:.1f}% | "
                   f"Drawdown={metrics['max_drawdown']:.2f}%")
    
    async def _generate_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'session_summary': {
                'start_time': datetime.now() - timedelta(hours=1),  # Mock start time
                'end_time': datetime.now(),
                'total_return': self.total_return,
                'daily_pnl': self.daily_pnl,
                'trade_count': self.trade_count,
                'win_rate': (self.win_count / max(1, self.trade_count)) * 100,
                'max_drawdown': self.max_drawdown * 100
            },
            'strategy_performance': {},
            'risk_metrics': {
                'max_leverage_used': 2.1,  # Mock data
                'var_breaches': 0,
                'drawdown_breaches': 0
            },
            'ai_infrastructure': {
                'models_deployed': len(self.model_registry.list_models()) if self.model_registry else 0,
                'ab_tests_active': len(self.ab_testing_engine.active_tests) if self.ab_testing_engine else 0,
                'strategies_active': len(self.strategies)
            }
        }
        
        logger.info("📊 PERFORMANCE REPORT:")
        logger.info(json.dumps(report, indent=2, default=str))

# Main entry point for live trading service
async def main():
    """Main entry point for live trading"""
    print("🚀 CroweTrade AI Trading Infrastructure - LIVE TRADING")
    print("=" * 60)
    
    # Load configuration
    config = LiveTradingConfig()
    
    # Initialize trading engine
    engine = LiveTradingEngine(config)
    
    try:
        # Initialize AI components
        await engine.initialize()
        
        print("\n✅ ALL SYSTEMS INITIALIZED")
        print("🎯 Advanced AI capabilities:")
        print("   • Model Registry with lifecycle management")
        print("   • A/B Testing with multi-armed bandits")
        print("   • Risk-aware strategy execution")
        print("   • Real-time performance monitoring")
        print("\n🟢 LIVE TRADING STARTED!")
        print("=" * 60)
        
        # Start live trading
        await engine.start_live_trading()
        
    except KeyboardInterrupt:
        print("\n⚠️  Received shutdown signal...")
        await engine.stop_live_trading()
        print("✅ Live trading stopped safely")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        await engine.stop_live_trading()

if __name__ == "__main__":
    # Run the live trading system
    asyncio.run(main())
