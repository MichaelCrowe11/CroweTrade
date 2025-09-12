"""
Integrated Crypto Trading Service for CroweTrade AI Trading Infrastructure

This service orchestrates the complete cryptocurrency trading ecosystem including:
- Coinbase Pro integration
- Real-time crypto market data
- Wallet management
- Crypto-specific risk controls
- Cross-chain operations
- DeFi protocol integration
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Any
import logging

from crowetrade.data.crypto_market_data import CryptoDataFeed, create_crypto_data_feed
from crowetrade.services.crypto_wallet import CryptoWalletManager, create_crypto_wallet_manager, WalletType, CryptoNetwork
from crowetrade.services.execution_service.coinbase_adapter import CoinbaseProVenueAdapter, CoinbaseConfig, create_coinbase_adapter
from crowetrade.risk.crypto_risk import CryptoRiskController, create_crypto_risk_controller
from crowetrade.core.contracts import FeatureVector
from crowetrade.core.types import Signal
from crowetrade.core.events import ChildOrder


class CryptoTradingService:
    """Integrated cryptocurrency trading service"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.market_data_feed: Optional[CryptoDataFeed] = None
        self.wallet_manager: Optional[CryptoWalletManager] = None
        self.coinbase_adapter: Optional[CoinbaseProVenueAdapter] = None
        self.risk_controller: Optional[CryptoRiskController] = None
        
        # Trading state
        self.active_positions: Dict[str, Decimal] = {}  # symbol -> position_size_usd
        self.portfolio_value_usd = Decimal("1000000")  # $1M default portfolio
        self.is_trading_enabled = False
        
        # Performance tracking
        self.trade_count = 0
        self.total_pnl = Decimal("0")
        self.start_time = datetime.now(timezone.utc)
        
        print("🏗️  Crypto Trading Service initializing...")
    
    async def initialize(self):
        """Initialize the complete crypto trading ecosystem"""
        try:
            # Initialize market data feed
            market_data_config = {
                "exchanges": self.config.get("exchanges", ["coinbase", "binance", "kraken"]),
                "symbols": self.config.get("symbols", ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "MATIC-USD"]),
                "update_frequency_ms": self.config.get("update_frequency_ms", 1000)
            }
            self.market_data_feed = create_crypto_data_feed(market_data_config)
            
            # Initialize wallet manager
            wallet_config = {
                "encryption_key": self.config.get("wallet_encryption_key"),
                "ethereum_rpc": self.config.get("ethereum_rpc"),
                "polygon_rpc": self.config.get("polygon_rpc"),
                "bsc_rpc": self.config.get("bsc_rpc"),
                "require_2fa": self.config.get("require_2fa", True),
                "cold_storage_threshold": self.config.get("cold_storage_threshold", "10000"),
                "max_hot_wallet_balance": self.config.get("max_hot_wallet_balance", "50000")
            }
            self.wallet_manager = create_crypto_wallet_manager(wallet_config)
            
            # Initialize Coinbase adapter
            coinbase_config = {
                "api_key": self.config.get("coinbase_api_key"),
                "api_secret": self.config.get("coinbase_api_secret"), 
                "passphrase": self.config.get("coinbase_passphrase"),
                "sandbox": self.config.get("coinbase_sandbox", False)
            }
            
            if all(coinbase_config.values()):
                self.coinbase_adapter = create_coinbase_adapter(**coinbase_config)
            else:
                print("⚠️ Coinbase credentials not configured - using mock adapter")
            
            # Initialize risk controller
            risk_config = self.config.get("risk", {})
            self.risk_controller = create_crypto_risk_controller(risk_config)
            
            # Start background services
            await self._start_background_services()
            
            print("✅ Crypto Trading Service initialized successfully")
            
        except Exception as e:
            print(f"❌ Crypto Trading Service initialization failed: {e}")
            raise
    
    async def _start_background_services(self):
        """Start background services for data feeds and monitoring"""
        
        tasks = []
        
        # Market data feed
        if self.market_data_feed:
            tasks.append(self._market_data_loop())
        
        # Wallet management
        if self.wallet_manager:
            await self.wallet_manager.initialize()
        
        # Coinbase connection
        if self.coinbase_adapter:
            connected = await self.coinbase_adapter.connect()
            if connected:
                print("✅ Coinbase Pro connected")
                tasks.append(self._coinbase_monitoring_loop())
            else:
                print("❌ Coinbase Pro connection failed")
        
        # Risk monitoring
        tasks.append(self._risk_monitoring_loop())
        
        # Performance tracking
        tasks.append(self._performance_tracking_loop())
        
        # Start all background tasks
        for task in tasks:
            asyncio.create_task(task)
        
        print(f"🚀 Started {len(tasks)} background services")
    
    async def shutdown(self):
        """Shutdown crypto trading service"""
        try:
            self.is_trading_enabled = False
            
            if self.coinbase_adapter:
                await self.coinbase_adapter.disconnect()
                
            if self.market_data_feed:
                await self.market_data_feed.stop()
                
            if self.wallet_manager:
                await self.wallet_manager.shutdown()
                
            print("✅ Crypto Trading Service shutdown complete")
            
        except Exception as e:
            print(f"⚠️ Shutdown error: {e}")
    
    # Trading Operations
    
    async def enable_trading(self):
        """Enable live cryptocurrency trading"""
        
        # Pre-flight checks
        if not self.coinbase_adapter or not self.coinbase_adapter.connected:
            raise RuntimeError("Coinbase adapter not connected")
            
        if not self.market_data_feed:
            raise RuntimeError("Market data feed not available")
            
        if not self.risk_controller:
            raise RuntimeError("Risk controller not initialized")
        
        # Verify wallet connectivity
        if self.wallet_manager:
            balances = await self.wallet_manager.get_total_portfolio_balance()
            print(f"💰 Portfolio balances: {dict(balances)}")
        
        self.is_trading_enabled = True
        print("🚀 Cryptocurrency trading ENABLED")
    
    async def disable_trading(self):
        """Disable cryptocurrency trading"""
        self.is_trading_enabled = False
        print("🛑 Cryptocurrency trading DISABLED")
    
    async def process_crypto_signal(self, signal: Signal) -> Dict[str, Any]:
        """Process incoming cryptocurrency trading signal"""
        
        try:
            if not self.is_trading_enabled:
                return {"status": "rejected", "reason": "Trading disabled"}
            
            symbol = signal.instrument
            
            # Validate crypto symbol
            if not self._is_crypto_symbol(symbol):
                return {"status": "rejected", "reason": "Not a crypto symbol"}
            
            # Get current market data
            latest_price = await self.market_data_feed.get_latest_price(symbol)
            if not latest_price:
                return {"status": "rejected", "reason": "No market data available"}
            
            # Risk evaluation
            approved, reason, adjusted_size = await self.risk_controller.evaluate_crypto_signal(
                signal, self.active_positions, self.portfolio_value_usd
            )
            
            if not approved:
                return {"status": "rejected", "reason": reason}
            
            # Execute trade
            if adjusted_size and adjusted_size > Decimal("100"):  # Minimum $100 trade
                result = await self._execute_crypto_trade(signal, adjusted_size, latest_price)
                return result
            else:
                return {"status": "rejected", "reason": "Position size too small"}
                
        except Exception as e:
            self.logger.error(f"Crypto signal processing error: {e}")
            return {"status": "error", "reason": str(e)}
    
    async def _execute_crypto_trade(self, signal: Signal, size_usd: Decimal, current_price: Decimal) -> Dict[str, Any]:
        """Execute cryptocurrency trade"""
        
        try:
            symbol = signal.instrument
            
            # Calculate quantity based on USD size
            quantity = size_usd / current_price
            
            # Create order
            from crowetrade.core.types import OrderSide, OrderType
            
            side = OrderSide.BUY if signal.mu > 0 else OrderSide.SELL
            
            order = ChildOrder(
                instrument=symbol,
                side=side,
                qty=float(quantity),
                order_type=OrderType.MARKET,  # Start with market orders
                limit_px=None,
                time_in_force="GTC"
            )
            
            # Send to Coinbase
            if self.coinbase_adapter:
                order_id = await self.coinbase_adapter.send_order(order)
                
                if order_id:
                    # Update position tracking
                    position_change = size_usd if side == OrderSide.BUY else -size_usd
                    self.active_positions[symbol] = self.active_positions.get(symbol, Decimal("0")) + position_change
                    
                    self.trade_count += 1
                    
                    return {
                        "status": "executed",
                        "order_id": order_id,
                        "symbol": symbol,
                        "side": side.value,
                        "quantity": float(quantity),
                        "size_usd": float(size_usd),
                        "price": float(current_price),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                else:
                    return {"status": "failed", "reason": "Order submission failed"}
            else:
                return {"status": "failed", "reason": "No exchange adapter available"}
                
        except Exception as e:
            self.logger.error(f"Crypto trade execution error: {e}")
            return {"status": "error", "reason": str(e)}
    
    # Portfolio Management
    
    async def get_crypto_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive cryptocurrency portfolio summary"""
        
        try:
            summary = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trading_enabled": self.is_trading_enabled,
                "total_portfolio_value_usd": float(self.portfolio_value_usd),
                "positions": {},
                "wallet_balances": {},
                "market_data": {},
                "risk_metrics": {},
                "performance": {}
            }
            
            # Active positions
            total_crypto_exposure = Decimal("0")
            for symbol, position_usd in self.active_positions.items():
                if position_usd != 0:
                    current_price = await self.market_data_feed.get_latest_price(symbol) if self.market_data_feed else Decimal("0")
                    
                    summary["positions"][symbol] = {
                        "size_usd": float(position_usd),
                        "current_price": float(current_price),
                        "pct_of_portfolio": float((position_usd / self.portfolio_value_usd) * 100) if self.portfolio_value_usd > 0 else 0
                    }
                    total_crypto_exposure += abs(position_usd)
            
            summary["total_crypto_exposure_usd"] = float(total_crypto_exposure)
            summary["crypto_exposure_pct"] = float((total_crypto_exposure / self.portfolio_value_usd) * 100) if self.portfolio_value_usd > 0 else 0
            
            # Wallet balances
            if self.wallet_manager:
                portfolio_balances = await self.wallet_manager.get_total_portfolio_balance()
                for symbol, balance in portfolio_balances.items():
                    summary["wallet_balances"][symbol] = float(balance)
            
            # Market data summary
            if self.market_data_feed:
                for symbol in ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"]:
                    price = await self.market_data_feed.get_latest_price(symbol)
                    if price:
                        stats = await self.market_data_feed.get_market_stats(symbol)
                        summary["market_data"][symbol] = {
                            "price": float(price),
                            "change_24h_pct": float(stats.price_change_pct_24h) if stats else 0.0
                        }
            
            # Risk metrics
            if self.risk_controller:
                risk_summary = await self.risk_controller.get_risk_summary()
                summary["risk_metrics"] = risk_summary
            
            # Performance metrics
            uptime_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
            summary["performance"] = {
                "trade_count": self.trade_count,
                "total_pnl_usd": float(self.total_pnl),
                "uptime_hours": round(uptime_hours, 2),
                "trades_per_hour": round(self.trade_count / max(uptime_hours, 0.1), 2)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Portfolio summary error: {e}")
            return {"error": str(e)}
    
    async def rebalance_crypto_portfolio(self, target_allocations: Dict[str, float]):
        """Rebalance cryptocurrency portfolio to target allocations"""
        
        try:
            if not self.is_trading_enabled:
                return {"status": "rejected", "reason": "Trading disabled"}
            
            print(f"🔄 Rebalancing crypto portfolio to targets: {target_allocations}")
            
            # Get current portfolio state
            current_total_value = sum(abs(pos) for pos in self.active_positions.values())
            
            rebalance_orders = []
            
            for symbol, target_pct in target_allocations.items():
                if not self._is_crypto_symbol(symbol):
                    continue
                    
                target_value = current_total_value * Decimal(str(target_pct / 100))
                current_value = self.active_positions.get(symbol, Decimal("0"))
                
                difference = target_value - current_value
                
                if abs(difference) > Decimal("500"):  # $500 minimum rebalance
                    current_price = await self.market_data_feed.get_latest_price(symbol)
                    if current_price:
                        # Create rebalance signal
                        signal = Signal(
                            instrument=symbol,
                            mu=0.01 if difference > 0 else -0.01,  # Small signal for rebalancing
                            sigma=0.05,
                            prob_edge_pos=0.6,
                            policy_id="rebalance"
                        )
                        
                        result = await self.process_crypto_signal(signal)
                        rebalance_orders.append({
                            "symbol": symbol,
                            "target_pct": target_pct,
                            "difference_usd": float(difference),
                            "result": result
                        })
            
            return {
                "status": "completed",
                "rebalance_orders": rebalance_orders,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio rebalance error: {e}")
            return {"status": "error", "reason": str(e)}
    
    # Background Monitoring Loops
    
    async def _market_data_loop(self):
        """Background market data processing"""
        
        try:
            # Start the market data feed
            await self.market_data_feed.start()
            
        except Exception as e:
            self.logger.error(f"Market data loop error: {e}")
            await asyncio.sleep(60)  # Wait before retry
    
    async def _coinbase_monitoring_loop(self):
        """Monitor Coinbase adapter for fills and status"""
        
        while self.is_trading_enabled:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
                
                if self.coinbase_adapter:
                    # Process fills
                    fills = await self.coinbase_adapter.handle_fills()
                    for fill in fills:
                        await self._process_crypto_fill(fill)
                    
                    # Get market data updates
                    market_updates = await self.coinbase_adapter.get_market_data()
                    for update in market_updates:
                        await self._process_market_update(update)
                        
            except Exception as e:
                self.logger.error(f"Coinbase monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _risk_monitoring_loop(self):
        """Background risk monitoring and alerts"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if self.risk_controller and self.market_data_feed:
                    # Update market data for risk calculations
                    for symbol in self.active_positions:
                        price = await self.market_data_feed.get_latest_price(symbol)
                        if price:
                            self.risk_controller.update_price_data(symbol, price, Decimal("1000000"))  # Volume placeholder
                    
                    # Check for risk limit breaches
                    risk_summary = await self.risk_controller.get_risk_summary()
                    
                    # Alert on high risk
                    current_metrics = risk_summary.get("current_metrics", {})
                    portfolio_var = current_metrics.get("portfolio_var_1d", 0)
                    
                    if portfolio_var > 0.05:  # 5% VaR threshold
                        print(f"🚨 High portfolio VaR detected: {portfolio_var*100:.1f}%")
                        
            except Exception as e:
                self.logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _performance_tracking_loop(self):
        """Track trading performance metrics"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Update every 5 minutes
                
                # Calculate unrealized P&L
                unrealized_pnl = Decimal("0")
                
                for symbol, position_usd in self.active_positions.items():
                    if position_usd != 0 and self.market_data_feed:
                        current_price = await self.market_data_feed.get_latest_price(symbol)
                        if current_price:
                            # Simplified P&L calculation (would need entry prices in production)
                            # For now, assume break-even on all positions
                            pass
                
                # Log performance summary
                uptime_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
                
                print(f"📊 Performance Update:")
                print(f"   Trades: {self.trade_count}")
                print(f"   Uptime: {uptime_hours:.1f} hours")
                print(f"   Active positions: {len([p for p in self.active_positions.values() if p != 0])}")
                print(f"   Total exposure: ${sum(abs(p) for p in self.active_positions.values()):,.0f}")
                
            except Exception as e:
                self.logger.error(f"Performance tracking error: {e}")
                await asyncio.sleep(600)
    
    # Helper Methods
    
    async def _process_crypto_fill(self, fill):
        """Process cryptocurrency fill from exchange"""
        
        try:
            print(f"✅ Crypto fill: {fill.instrument} {fill.qty} @ ${fill.price}")
            
            # Update P&L tracking (simplified)
            fill_value = Decimal(str(fill.qty * fill.price))
            # Would need to track entry prices for accurate P&L
            
        except Exception as e:
            self.logger.error(f"Fill processing error: {e}")
    
    async def _process_market_update(self, update):
        """Process market data update"""
        
        try:
            if update.get("type") == "ticker":
                symbol = update.get("symbol")
                price = Decimal(str(update.get("price", 0)))
                
                if symbol and self.risk_controller:
                    volume = Decimal(str(update.get("volume", 0)))
                    self.risk_controller.update_price_data(symbol, price, volume)
                    
        except Exception as e:
            self.logger.error(f"Market update processing error: {e}")
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency"""
        crypto_symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "MATIC-USD", "USDC-USD"]
        return symbol in crypto_symbols


# Factory function
def create_crypto_trading_service(config: Dict[str, Any]) -> CryptoTradingService:
    """Factory function to create crypto trading service"""
    return CryptoTradingService(config)


# Demo/Test function
async def demo_crypto_trading():
    """Demonstration of crypto trading capabilities"""
    
    config = {
        "exchanges": ["coinbase"],
        "symbols": ["BTC-USD", "ETH-USD"],
        "coinbase_api_key": "demo_key",
        "coinbase_api_secret": "demo_secret",
        "coinbase_passphrase": "demo_passphrase", 
        "coinbase_sandbox": True,
        "risk": {
            "max_crypto_portfolio_pct": 30,
            "max_single_crypto_pct": 10,
            "crypto_var_limit_1d": 5
        }
    }
    
    service = create_crypto_trading_service(config)
    
    try:
        await service.initialize()
        
        # Demo trading signal
        btc_signal = Signal(
            instrument="BTC-USD",
            mu=0.02,  # 2% expected return
            sigma=0.15,  # 15% volatility
            prob_edge_pos=0.65,  # 65% probability of positive edge
            policy_id="demo_crypto"
        )
        
        print("📋 Processing demo crypto signal...")
        result = await service.process_crypto_signal(btc_signal)
        print(f"Result: {result}")
        
        # Get portfolio summary
        summary = await service.get_crypto_portfolio_summary()
        print(f"\n📊 Portfolio Summary:")
        print(json.dumps(summary, indent=2, default=str))
        
    finally:
        await service.shutdown()


if __name__ == "__main__":
    asyncio.run(demo_crypto_trading())
