#!/usr/bin/env python3
"""
CroweTrade Crypto Trading Live Demo

This script demonstrates the complete cryptocurrency trading ecosystem:
- Real-time multi-exchange market data
- Advanced crypto wallet management
- Sophisticated risk controls
- Live Coinbase Pro trading execution
- Cross-chain bridge operations
- AI-driven trading signals

Usage:
    python demo_crypto_trading.py [--live] [--duration 3600]
"""

import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crowetrade.services.crypto_trading_service import create_crypto_trading_service
from crowetrade.core.types import Signal


class CryptoTradingDemo:
    """Live demonstration of crypto trading capabilities"""
    
    def __init__(self, live_mode: bool = False):
        self.live_mode = live_mode
        self.crypto_service = None
        
        print("🎯 CroweTrade Crypto Trading Live Demo")
        print("=" * 45)
        
        if live_mode:
            print("⚡ LIVE TRADING MODE - Real money at risk!")
        else:
            print("🛡️ DEMO MODE - Safe simulation environment")
        print()
    
    async def setup_demo_environment(self):
        """Setup the demo environment with realistic configuration"""
        
        demo_config = {
            # Market data configuration
            "exchanges": ["coinbase", "binance", "kraken"],
            "symbols": ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "MATIC-USD"],
            "update_frequency_ms": 1000,
            
            # Exchange APIs (demo vs live)
            "coinbase_api_key": "demo_key" if not self.live_mode else None,
            "coinbase_api_secret": "demo_secret" if not self.live_mode else None,
            "coinbase_passphrase": "demo_passphrase" if not self.live_mode else None,
            "coinbase_sandbox": not self.live_mode,
            
            # Wallet configuration
            "wallet_encryption_key": "demo_key_do_not_use_in_production",
            "ethereum_rpc": "https://eth-mainnet.alchemyapi.io/v2/demo",
            "polygon_rpc": "https://polygon-rpc.com",
            "bsc_rpc": "https://bsc-dataseed.binance.org",
            "require_2fa": False,  # Simplified for demo
            "cold_storage_threshold": "25000",  # $25k threshold
            "max_hot_wallet_balance": "100000",  # $100k max hot
            
            # Risk configuration
            "risk": {
                "max_crypto_portfolio_pct": 25.0,  # Conservative 25% crypto allocation
                "max_single_crypto_pct": 8.0,      # Max 8% per crypto
                "max_alt_crypto_pct": 3.0,         # Max 3% per altcoin
                "crypto_var_limit_1d": 4.0,        # 4% daily VaR limit
                "volatility_target": 0.18,          # 18% annual volatility target
                "max_daily_drawdown_pct": 1.5,     # 1.5% daily drawdown limit
                "kelly_fraction": 0.2,              # Conservative Kelly sizing
                "correlation_limit": 0.75           # Max 75% correlation
            }
        }
        
        # Override with real credentials in live mode
        if self.live_mode:
            import os
            demo_config["coinbase_api_key"] = os.getenv("COINBASE_API_KEY")
            demo_config["coinbase_api_secret"] = os.getenv("COINBASE_API_SECRET")
            demo_config["coinbase_passphrase"] = os.getenv("COINBASE_PASSPHRASE")
            
            if not all([demo_config["coinbase_api_key"], 
                       demo_config["coinbase_api_secret"], 
                       demo_config["coinbase_passphrase"]]):
                print("❌ Live mode requires COINBASE_* environment variables")
                return False
        
        # Initialize crypto trading service
        self.crypto_service = create_crypto_trading_service(demo_config)
        await self.crypto_service.initialize()
        
        return True
    
    async def demonstrate_market_data_feeds(self):
        """Demonstrate real-time crypto market data aggregation"""
        
        print("📊 Multi-Exchange Market Data Demonstration")
        print("-" * 45)
        
        if not self.crypto_service.market_data_feed:
            print("❌ Market data feed not available")
            return
        
        symbols = ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD"]
        
        for symbol in symbols:
            try:
                # Get latest price from consensus feed
                price = await self.crypto_service.market_data_feed.get_latest_price(symbol)
                
                # Get market statistics
                stats = await self.crypto_service.market_data_feed.get_market_stats(symbol)
                
                # Get technical indicators
                indicators = await self.crypto_service.market_data_feed.get_technical_indicators(symbol)
                
                print(f"\n{symbol}:")
                print(f"  💰 Price: ${price:,.2f}" if price else "  💰 Price: N/A")
                
                if stats:
                    print(f"  📈 24h Change: {stats.price_change_pct_24h:+.2f}%")
                    print(f"  📊 Volume 24h: ${stats.volume_24h:,.0f}")
                    print(f"  💎 Market Cap: ${stats.market_cap:,.0f}")
                
                if indicators:
                    print(f"  🔍 RSI: {indicators.get('rsi', 0):.1f}")
                    print(f"  📏 Volatility: {indicators.get('volatility', 0)*100:.1f}%")
                    print(f"  ⚖️ Momentum: {indicators.get('momentum', 0)*100:+.1f}%")
                
            except Exception as e:
                print(f"  ❌ Error getting data for {symbol}: {e}")
        
        print(f"\n✅ Market data demonstration complete")
    
    async def demonstrate_wallet_operations(self):
        """Demonstrate crypto wallet management capabilities"""
        
        print("\n🏦 Crypto Wallet Management Demonstration")
        print("-" * 45)
        
        if not self.crypto_service.wallet_manager:
            print("❌ Wallet manager not available")
            return
        
        try:
            # Get portfolio balances across all networks
            print("💰 Portfolio Balances:")
            balances = await self.crypto_service.wallet_manager.get_total_portfolio_balance()
            
            total_value_usd = 0
            for symbol, balance in balances.items():
                if balance > 0:
                    # Get current price for USD conversion
                    price_symbol = f"{symbol}-USD" if symbol != "USD" else "USDC-USD"
                    price = await self.crypto_service.market_data_feed.get_latest_price(price_symbol) if symbol != "USD" else 1
                    
                    if price:
                        value_usd = balance * price
                        total_value_usd += value_usd
                        print(f"  {symbol}: {balance:,.6f} (${value_usd:,.2f})")
                    else:
                        print(f"  {symbol}: {balance:,.6f}")
            
            print(f"\n  💎 Total Portfolio Value: ${total_value_usd:,.2f}")
            
            # Demonstrate hot/cold wallet monitoring
            print(f"\n🔥 Hot Wallet Status:")
            hot_balance = await self.crypto_service.wallet_manager.get_hot_wallet_balance()
            cold_threshold = float(self.crypto_service.wallet_manager.cold_storage_threshold)
            
            for network, balance_usd in hot_balance.items():
                status = "🔴 OVER LIMIT" if balance_usd > cold_threshold else "✅ Safe"
                print(f"  {network}: ${balance_usd:,.2f} {status}")
            
            # Show security features
            print(f"\n🛡️ Security Features:")
            print(f"  🔐 2FA Required: {self.crypto_service.wallet_manager.require_2fa}")
            print(f"  💎 Cold Storage Threshold: ${cold_threshold:,.0f}")
            print(f"  🔥 Max Hot Wallet: ${float(self.crypto_service.wallet_manager.max_hot_wallet_balance):,.0f}")
            print(f"  📱 Transaction Monitoring: Active")
            
            # Demonstrate cross-chain capabilities
            if hasattr(self.crypto_service.wallet_manager, 'bridge_manager'):
                print(f"\n🌉 Cross-Chain Bridge Status:")
                supported_chains = ["ethereum", "polygon", "bsc"]
                for chain in supported_chains:
                    try:
                        # Estimate bridge costs (demo)
                        estimate = await self.crypto_service.wallet_manager.estimate_bridge_cost(
                            from_network=chain,
                            to_network="ethereum",
                            amount_usd=1000
                        )
                        print(f"  {chain.upper()} → ETH: ${estimate.get('fee_usd', 0):.2f} "
                              f"({estimate.get('time_minutes', 0)} min)")
                    except:
                        print(f"  {chain.upper()}: Bridge offline")
            
        except Exception as e:
            print(f"❌ Wallet demonstration error: {e}")
    
    async def demonstrate_risk_controls(self):
        """Demonstrate advanced crypto risk management"""
        
        print("\n🛡️ Advanced Risk Management Demonstration")
        print("-" * 45)
        
        if not self.crypto_service.risk_controller:
            print("❌ Risk controller not available")
            return
        
        try:
            # Get comprehensive risk summary
            risk_summary = await self.crypto_service.risk_controller.get_risk_summary()
            
            print("📊 Portfolio Risk Metrics:")
            
            current_metrics = risk_summary.get("current_metrics", {})
            limits = risk_summary.get("limits", {})
            
            # VaR metrics
            var_1d = current_metrics.get("portfolio_var_1d", 0) * 100
            var_limit = limits.get("var_limit_1d_pct", 5)
            var_status = "🔴 BREACH" if var_1d > var_limit else "✅ Safe"
            print(f"  📉 1-Day VaR: {var_1d:.2f}% (limit: {var_limit}%) {var_status}")
            
            # Volatility metrics
            portfolio_vol = current_metrics.get("portfolio_volatility", 0) * 100
            vol_target = limits.get("volatility_target_pct", 18)
            vol_status = "⚠️ High" if portfolio_vol > vol_target * 1.5 else "✅ Normal"
            print(f"  📊 Portfolio Vol: {portfolio_vol:.1f}% (target: {vol_target}%) {vol_status}")
            
            # Concentration metrics
            max_position = current_metrics.get("max_position_pct", 0) * 100
            position_limit = limits.get("max_single_crypto_pct", 8)
            conc_status = "🔴 HIGH" if max_position > position_limit else "✅ Safe"
            print(f"  🎯 Max Position: {max_position:.1f}% (limit: {position_limit}%) {conc_status}")
            
            # Correlation analysis
            avg_correlation = current_metrics.get("avg_crypto_correlation", 0)
            corr_limit = limits.get("correlation_limit", 0.75)
            corr_status = "⚠️ High" if avg_correlation > corr_limit else "✅ Diversified"
            print(f"  🔗 Avg Correlation: {avg_correlation:.2f} (limit: {corr_limit}) {corr_status}")
            
            # Regime detection
            print(f"\n🎭 Market Regime Analysis:")
            regime_data = risk_summary.get("regime_analysis", {})
            current_regime = regime_data.get("current_regime", "unknown")
            regime_prob = regime_data.get("regime_probability", 0)
            
            regime_emoji = {
                "bull": "🟢", "bear": "🔴", "sideways": "🟡", "high_vol": "🟠"
            }.get(current_regime, "❓")
            
            print(f"  {regime_emoji} Current Regime: {current_regime.title()} ({regime_prob*100:.0f}% confidence)")
            
            volatility_spike = regime_data.get("volatility_spike_detected", False)
            if volatility_spike:
                print(f"  🚨 Volatility Spike Detected - Risk Limits Tightened")
            
            # Risk alerts
            alerts = risk_summary.get("active_alerts", [])
            if alerts:
                print(f"\n🚨 Active Risk Alerts:")
                for alert in alerts:
                    print(f"  ⚠️ {alert}")
            else:
                print(f"\n✅ No active risk alerts")
                
        except Exception as e:
            print(f"❌ Risk demonstration error: {e}")
    
    async def demonstrate_ai_trading_signals(self):
        """Demonstrate AI-driven crypto trading signals"""
        
        print("\n🤖 AI Trading Signal Demonstration")
        print("-" * 40)
        
        # Generate realistic demo signals based on market conditions
        demo_signals = [
            {
                "signal": Signal(
                    instrument="BTC-USD",
                    mu=0.012,  # 1.2% expected return
                    sigma=0.15,  # 15% volatility
                    prob_edge_pos=0.68,  # 68% edge probability
                    policy_id="momentum_btc_v1"
                ),
                "strategy": "Cross-Sectional Momentum",
                "description": "Bitcoin showing strong momentum vs. altcoins"
            },
            {
                "signal": Signal(
                    instrument="ETH-USD",
                    mu=0.025,  # 2.5% expected return
                    sigma=0.22,  # 22% volatility
                    prob_edge_pos=0.74,  # 74% edge probability
                    policy_id="mean_reversion_eth_v2"
                ),
                "strategy": "Mean Reversion",
                "description": "Ethereum oversold on technical indicators"
            },
            {
                "signal": Signal(
                    instrument="ADA-USD",
                    mu=-0.015,  # -1.5% expected return (short)
                    sigma=0.28,  # 28% volatility
                    prob_edge_pos=0.62,  # 62% edge probability
                    policy_id="trend_follow_ada_v1"
                ),
                "strategy": "Trend Following",
                "description": "Cardano breaking key support levels"
            }
        ]
        
        print("🎯 Processing AI-Generated Trading Signals:")
        print()
        
        for i, signal_data in enumerate(demo_signals, 1):
            signal = signal_data["signal"]
            strategy = signal_data["strategy"]
            description = signal_data["description"]
            
            print(f"{i}. {signal.instrument} - {strategy}")
            print(f"   📝 {description}")
            print(f"   📈 Expected Return: {signal.mu*100:+.1f}%")
            print(f"   📊 Volatility: {signal.sigma*100:.1f}%")
            print(f"   🎯 Edge Probability: {signal.prob_edge_pos*100:.0f}%")
            print(f"   🏷️  Policy ID: {signal.policy_id}")
            
            # Process the signal
            result = await self.crypto_service.process_crypto_signal(signal)
            
            status = result.get("status", "unknown")
            if status == "executed":
                print(f"   ✅ EXECUTED - Size: ${result.get('size_usd', 0):,.0f}")
                if result.get('order_id'):
                    print(f"      Order ID: {result['order_id']}")
            elif status == "rejected":
                reason = result.get("reason", "Unknown")
                print(f"   ❌ REJECTED - {reason}")
            elif status == "failed":
                print(f"   ⚠️ FAILED - {result.get('reason', 'Execution error')}")
            else:
                print(f"   ❓ {status.upper()}")
            
            print()
            
            # Brief pause between signals
            await asyncio.sleep(1)
        
        print("✅ AI signal processing demonstration complete")
    
    async def show_live_portfolio_dashboard(self):
        """Display live portfolio dashboard"""
        
        print("\n📊 Live Portfolio Dashboard")
        print("=" * 30)
        
        try:
            summary = await self.crypto_service.get_crypto_portfolio_summary()
            
            # Header
            timestamp = summary.get("timestamp", "")
            trading_status = "🟢 ACTIVE" if summary.get("trading_enabled") else "🔴 INACTIVE"
            print(f"Status: {trading_status} | Time: {timestamp[:19]}")
            print("-" * 50)
            
            # Portfolio overview
            total_value = summary.get("total_portfolio_value_usd", 0)
            crypto_exposure = summary.get("crypto_exposure_pct", 0)
            
            print(f"💰 Total Portfolio: ${total_value:,.0f}")
            print(f"🪙 Crypto Exposure: {crypto_exposure:.1f}%")
            
            # Active positions
            positions = summary.get("positions", {})
            if positions:
                print(f"\n📈 Active Positions:")
                for symbol, pos_data in positions.items():
                    if pos_data["size_usd"] != 0:
                        size_usd = pos_data["size_usd"]
                        pct_portfolio = pos_data["pct_of_portfolio"]
                        current_price = pos_data["current_price"]
                        
                        size_emoji = "🟢" if size_usd > 0 else "🔴"
                        print(f"  {size_emoji} {symbol}: ${abs(size_usd):,.0f} "
                              f"({pct_portfolio:.1f}%) @ ${current_price:,.2f}")
            else:
                print(f"\n📈 No active positions")
            
            # Market prices
            market_data = summary.get("market_data", {})
            if market_data:
                print(f"\n💹 Market Prices:")
                for symbol, data in market_data.items():
                    price = data.get("price", 0)
                    change = data.get("change_24h_pct", 0)
                    change_emoji = "📈" if change >= 0 else "📉"
                    print(f"  {change_emoji} {symbol}: ${price:,.2f} ({change:+.1f}%)")
            
            # Performance metrics
            performance = summary.get("performance", {})
            if performance:
                print(f"\n🏆 Performance:")
                print(f"  📊 Total Trades: {performance.get('trade_count', 0)}")
                print(f"  💰 Total P&L: ${performance.get('total_pnl_usd', 0):+,.0f}")
                print(f"  ⏱️ Uptime: {performance.get('uptime_hours', 0):.1f} hours")
                print(f"  ⚡ Trade Rate: {performance.get('trades_per_hour', 0):.1f}/hour")
            
            # Risk metrics
            risk_metrics = summary.get("risk_metrics", {})
            if risk_metrics and risk_metrics.get("current_metrics"):
                current = risk_metrics["current_metrics"]
                print(f"\n🛡️ Risk Metrics:")
                print(f"  📉 Portfolio VaR: {current.get('portfolio_var_1d', 0)*100:.1f}%")
                print(f"  📊 Volatility: {current.get('portfolio_volatility', 0)*100:.1f}%")
                print(f"  🎯 Max Position: {current.get('max_position_pct', 0)*100:.1f}%")
            
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
    
    async def run_demo_session(self, duration_seconds: int = 300):
        """Run a complete demo session"""
        
        print(f"\n🎬 Starting Demo Session ({duration_seconds//60} minutes)")
        print("=" * 50)
        
        try:
            # Setup
            success = await self.setup_demo_environment()
            if not success:
                return
            
            print("✅ Demo environment initialized")
            print()
            
            # Enable trading if in live mode
            if self.live_mode:
                await self.crypto_service.enable_trading()
            
            # Run demonstrations
            await self.demonstrate_market_data_feeds()
            await self.demonstrate_wallet_operations()
            await self.demonstrate_risk_controls()
            await self.demonstrate_ai_trading_signals()
            
            # Show dashboard
            await self.show_live_portfolio_dashboard()
            
            # Monitor for specified duration
            print(f"\n⏰ Monitoring trading session for {duration_seconds//60} minutes...")
            print("Press Ctrl+C to stop early")
            
            end_time = asyncio.get_event_loop().time() + duration_seconds
            
            while asyncio.get_event_loop().time() < end_time:
                await asyncio.sleep(30)  # Update every 30 seconds
                
                # Quick status update
                summary = await self.crypto_service.get_crypto_portfolio_summary()
                performance = summary.get("performance", {})
                
                print(f"\r⏰ Running... | "
                      f"Trades: {performance.get('trade_count', 0)} | "
                      f"Exposure: {summary.get('crypto_exposure_pct', 0):.1f}% | "
                      f"P&L: ${performance.get('total_pnl_usd', 0):+,.0f}", end="")
            
            print(f"\n\n✅ Demo session completed successfully!")
            
            # Final dashboard
            print(f"\n📊 Final Portfolio State:")
            await self.show_live_portfolio_dashboard()
            
        except KeyboardInterrupt:
            print(f"\n\n⏹️ Demo session interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Demo session error: {e}")
        
        finally:
            # Cleanup
            if self.crypto_service:
                await self.crypto_service.shutdown()
            print(f"\n🏁 Demo session ended")


async def main():
    """Main entry point for crypto trading demo"""
    
    parser = argparse.ArgumentParser(description="CroweTrade Crypto Trading Demo")
    parser.add_argument("--live", action="store_true", 
                       help="Run with live trading (requires API credentials)")
    parser.add_argument("--duration", type=int, default=300,
                       help="Demo duration in seconds (default: 300)")
    
    args = parser.parse_args()
    
    if args.live:
        print("⚠️ WARNING: Live mode will use real API credentials and may execute actual trades!")
        response = input("Are you sure you want to proceed? (YES/no): ")
        if response.upper() != "YES":
            print("❌ Live demo cancelled")
            return
    
    demo = CryptoTradingDemo(live_mode=args.live)
    await demo.run_demo_session(duration_seconds=args.duration)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
