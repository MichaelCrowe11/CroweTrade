#!/usr/bin/env python3
"""
CroweTrade Crypto Trading Activation Script

This script activates comprehensive cryptocurrency trading capabilities including:
- Coinbase Pro venue adapter
- Multi-exchange crypto market data
- Crypto wallet management
- Crypto-specific risk controls

Usage:
    python activate_crypto_trading.py [--demo] [--config crypto_trading.yaml]
"""

import asyncio
import os
import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime, timezone

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crowetrade.services.crypto_trading_service import create_crypto_trading_service
from crowetrade.core.types import Signal


class CryptoTradingActivator:
    """Activates and manages cryptocurrency trading capabilities"""
    
    def __init__(self, config_path: str = None, demo_mode: bool = False):
        self.demo_mode = demo_mode
        self.config_path = config_path or "config/crypto_trading.yaml"
        self.config = {}
        self.crypto_service = None
        
        print("🚀 CroweTrade Crypto Trading Activator")
        print("=" * 50)
    
    async def load_configuration(self):
        """Load crypto trading configuration"""
        
        try:
            with open(self.config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            self.config = full_config.get('crypto_trading', {})
            
            if not self.config.get('enabled', False):
                raise RuntimeError("Crypto trading is disabled in configuration")
            
            print(f"✅ Loaded configuration from {self.config_path}")
            
            # Override with environment variables for sensitive data
            if not self.demo_mode:
                self.config['coinbase_api_key'] = os.getenv('COINBASE_API_KEY')
                self.config['coinbase_api_secret'] = os.getenv('COINBASE_API_SECRET')
                self.config['coinbase_passphrase'] = os.getenv('COINBASE_PASSPHRASE')
                
                missing_creds = []
                if not self.config['coinbase_api_key']:
                    missing_creds.append('COINBASE_API_KEY')
                if not self.config['coinbase_api_secret']:
                    missing_creds.append('COINBASE_API_SECRET')
                if not self.config['coinbase_passphrase']:
                    missing_creds.append('COINBASE_PASSPHRASE')
                
                if missing_creds:
                    print(f"⚠️ Missing environment variables: {', '.join(missing_creds)}")
                    print("ℹ️ Running in demo mode without live trading")
                    self.demo_mode = True
            
            if self.demo_mode:
                self.config['coinbase_sandbox'] = True
                print("🎯 Demo mode activated")
            
        except Exception as e:
            print(f"❌ Configuration loading failed: {e}")
            raise
    
    async def initialize_crypto_service(self):
        """Initialize the crypto trading service"""
        
        try:
            self.crypto_service = create_crypto_trading_service(self.config)
            await self.crypto_service.initialize()
            
            print("✅ Crypto trading service initialized")
            
        except Exception as e:
            print(f"❌ Crypto service initialization failed: {e}")
            raise
    
    async def run_system_checks(self):
        """Run comprehensive system checks"""
        
        print("\n🔍 Running System Checks")
        print("-" * 30)
        
        checks = []
        
        # Configuration check
        if self.config:
            checks.append(("Configuration", "✅ Loaded"))
        else:
            checks.append(("Configuration", "❌ Missing"))
        
        # Service check
        if self.crypto_service:
            checks.append(("Crypto Service", "✅ Initialized"))
        else:
            checks.append(("Crypto Service", "❌ Failed"))
        
        # Exchange connectivity
        if hasattr(self.crypto_service, 'coinbase_adapter') and self.crypto_service.coinbase_adapter:
            if hasattr(self.crypto_service.coinbase_adapter, 'connected') and self.crypto_service.coinbase_adapter.connected:
                checks.append(("Coinbase Pro", "✅ Connected"))
            else:
                checks.append(("Coinbase Pro", "⚠️ Disconnected"))
        else:
            checks.append(("Coinbase Pro", "⚠️ Not configured"))
        
        # Market data feed
        if hasattr(self.crypto_service, 'market_data_feed') and self.crypto_service.market_data_feed:
            checks.append(("Market Data", "✅ Active"))
        else:
            checks.append(("Market Data", "❌ Inactive"))
        
        # Risk controller
        if hasattr(self.crypto_service, 'risk_controller') and self.crypto_service.risk_controller:
            checks.append(("Risk Controls", "✅ Active"))
        else:
            checks.append(("Risk Controls", "❌ Inactive"))
        
        # Wallet manager
        if hasattr(self.crypto_service, 'wallet_manager') and self.crypto_service.wallet_manager:
            checks.append(("Wallet Manager", "✅ Active"))
        else:
            checks.append(("Wallet Manager", "❌ Inactive"))
        
        for check_name, status in checks:
            print(f"{check_name:.<20} {status}")
        
        # Overall health
        failed_checks = [check for check, status in checks if "❌" in status]
        
        if failed_checks:
            print(f"\n⚠️ {len(failed_checks)} checks failed")
            return False
        else:
            print(f"\n✅ All {len(checks)} checks passed")
            return True
    
    async def demonstrate_crypto_capabilities(self):
        """Demonstrate cryptocurrency trading capabilities"""
        
        print("\n🎯 Demonstrating Crypto Capabilities")
        print("-" * 40)
        
        try:
            # Test crypto signal processing
            test_signals = [
                Signal(
                    instrument="BTC-USD",
                    mu=0.015,
                    sigma=0.12,
                    prob_edge_pos=0.65,
                    policy_id="demo_btc"
                ),
                Signal(
                    instrument="ETH-USD", 
                    mu=0.025,
                    sigma=0.18,
                    prob_edge_pos=0.72,
                    policy_id="demo_eth"
                ),
                Signal(
                    instrument="ADA-USD",
                    mu=0.035,
                    sigma=0.25,
                    prob_edge_pos=0.68,
                    policy_id="demo_ada"
                )
            ]
            
            print("📊 Processing demo crypto signals...")
            
            for i, signal in enumerate(test_signals, 1):
                print(f"\n{i}. Processing {signal.instrument} signal:")
                print(f"   Expected return: {signal.mu*100:.1f}%")
                print(f"   Volatility: {signal.sigma*100:.1f}%")
                print(f"   Edge probability: {signal.prob_edge_pos*100:.1f}%")
                
                result = await self.crypto_service.process_crypto_signal(signal)
                
                status = result.get('status', 'unknown')
                if status == 'executed':
                    print(f"   ✅ EXECUTED - Order ID: {result.get('order_id', 'N/A')}")
                    print(f"      Size: ${result.get('size_usd', 0):,.0f}")
                    print(f"      Price: ${result.get('price', 0):,.2f}")
                elif status == 'rejected':
                    print(f"   ⚠️ REJECTED - {result.get('reason', 'Unknown reason')}")
                else:
                    print(f"   ❓ {status.upper()} - {result.get('reason', 'No details')}")
            
            # Get portfolio summary
            print(f"\n📈 Portfolio Summary:")
            summary = await self.crypto_service.get_crypto_portfolio_summary()
            
            print(f"   Trading enabled: {summary.get('trading_enabled', False)}")
            print(f"   Total portfolio: ${summary.get('total_portfolio_value_usd', 0):,.0f}")
            print(f"   Crypto exposure: {summary.get('crypto_exposure_pct', 0):.1f}%")
            print(f"   Trade count: {summary.get('performance', {}).get('trade_count', 0)}")
            
            positions = summary.get('positions', {})
            if positions:
                print(f"   Active positions:")
                for symbol, pos_data in positions.items():
                    if pos_data['size_usd'] != 0:
                        print(f"     {symbol}: ${pos_data['size_usd']:,.0f} ({pos_data['pct_of_portfolio']:.1f}%)")
            
            # Market data summary
            market_data = summary.get('market_data', {})
            if market_data:
                print(f"   Market prices:")
                for symbol, data in market_data.items():
                    change = data.get('change_24h_pct', 0)
                    change_str = f"+{change:.1f}%" if change >= 0 else f"{change:.1f}%"
                    print(f"     {symbol}: ${data.get('price', 0):,.2f} ({change_str})")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise
    
    async def activate_live_trading(self):
        """Activate live cryptocurrency trading"""
        
        if self.demo_mode:
            print("\n🎯 Demo mode - live trading not activated")
            return
        
        print("\n🚀 Activating Live Crypto Trading")
        print("-" * 35)
        
        try:
            # Final safety check
            response = input("Are you sure you want to enable live cryptocurrency trading? (YES/no): ")
            if response.upper() != "YES":
                print("❌ Live trading activation cancelled")
                return
            
            await self.crypto_service.enable_trading()
            
            print("✅ Live cryptocurrency trading ACTIVATED")
            print("⚠️ Monitor positions carefully")
            print("📞 Support: Use kill switch if needed")
            
        except Exception as e:
            print(f"❌ Live trading activation failed: {e}")
            raise
    
    async def monitor_trading_session(self, duration_minutes: int = 60):
        """Monitor active trading session"""
        
        print(f"\n📊 Monitoring Trading Session ({duration_minutes} minutes)")
        print("-" * 50)
        
        start_time = datetime.now(timezone.utc)
        end_time = start_time.timestamp() + (duration_minutes * 60)
        
        try:
            while datetime.now(timezone.utc).timestamp() < end_time:
                # Get current status
                summary = await self.crypto_service.get_crypto_portfolio_summary()
                
                # Display key metrics
                print(f"\r⏰ {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')} | "
                      f"Trades: {summary.get('performance', {}).get('trade_count', 0)} | "
                      f"Exposure: {summary.get('crypto_exposure_pct', 0):.1f}% | "
                      f"P&L: ${summary.get('performance', {}).get('total_pnl_usd', 0):+,.0f}", end="")
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
        except KeyboardInterrupt:
            print(f"\n\n⏹️ Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")
    
    async def shutdown_gracefully(self):
        """Shutdown crypto trading service gracefully"""
        
        print("\n🛑 Shutting down crypto trading service...")
        
        try:
            if self.crypto_service:
                await self.crypto_service.shutdown()
            print("✅ Shutdown complete")
            
        except Exception as e:
            print(f"⚠️ Shutdown error: {e}")
    
    async def run_full_activation(self, monitor_duration: int = 0):
        """Run complete crypto trading activation process"""
        
        try:
            # Load configuration
            await self.load_configuration()
            
            # Initialize service
            await self.initialize_crypto_service()
            
            # Run system checks
            checks_passed = await self.run_system_checks()
            
            if not checks_passed:
                print("❌ System checks failed - aborting activation")
                return
            
            # Demonstrate capabilities
            await self.demonstrate_crypto_capabilities()
            
            # Activate live trading (if not demo)
            if not self.demo_mode:
                await self.activate_live_trading()
            
            # Monitor session if requested
            if monitor_duration > 0:
                await self.monitor_trading_session(monitor_duration)
            
            print(f"\n🎉 Crypto trading activation complete!")
            
            if not self.demo_mode and self.crypto_service and self.crypto_service.is_trading_enabled:
                print("⚡ Live cryptocurrency trading is now ACTIVE")
                print("📊 Monitor dashboard: http://localhost:3000")
                print("🛑 Emergency stop: Ctrl+C or kill switch")
            
        except Exception as e:
            print(f"❌ Activation failed: {e}")
            raise
        
        finally:
            await self.shutdown_gracefully()


async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Activate CroweTrade cryptocurrency trading")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode")
    parser.add_argument("--config", default="config/crypto_trading.yaml", help="Configuration file path")
    parser.add_argument("--monitor", type=int, default=0, help="Monitor duration in minutes")
    
    args = parser.parse_args()
    
    activator = CryptoTradingActivator(
        config_path=args.config,
        demo_mode=args.demo
    )
    
    await activator.run_full_activation(monitor_duration=args.monitor)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Crypto trading activation interrupted")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
