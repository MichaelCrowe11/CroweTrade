#!/usr/bin/env python3
"""
CroweTrade Live Trading Demo - Production Launch

This script demonstrates the live trading capabilities using the advanced AI infrastructure:
- Model Registry for strategy management
- A/B Testing for model optimization  
- Real-time risk management
- Performance monitoring

⚠️  DEMO MODE: Uses simulated market data and paper trading
"""

import asyncio
import signal
import sys
from datetime import datetime
from crowetrade.services.live_trading import LiveTradingEngine, LiveTradingConfig

def signal_handler(signum, frame):
    """Handle graceful shutdown"""
    print("\n🛑 Graceful shutdown initiated...")
    sys.exit(0)

async def main():
    """Run live trading demo"""
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 CroweTrade Advanced AI Trading Infrastructure")
    print("=" * 60)
    print("🎯 LIVE TRADING DEMO - Production Capabilities")
    print("=" * 60)
    print(f"⏰ Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("🔧 Configuration:")
    config = LiveTradingConfig()
    print(f"   • Max Position Size: ${config.max_position_size:,.0f}")
    print(f"   • Max Daily Loss: ${config.max_daily_loss:,.0f}")
    print(f"   • Risk Check Frequency: {config.risk_check_frequency}s")
    print(f"   • Model Evaluation: {config.model_evaluation_frequency}s")
    print(f"   • Rebalancing: {config.rebalance_frequency}s")
    print()
    
    print("🤖 AI Infrastructure:")
    print("   • Model Registry with lifecycle management")
    print("   • A/B Testing with multi-armed bandits")
    print("   • Real-time risk monitoring")
    print("   • Automated strategy execution")
    print("   • Performance analytics")
    print()
    
    print("⚠️  DEMO MODE: Paper trading with simulated data")
    print("🟢 Ready to start live trading...")
    print()
    
    try:
        # Initialize and run live trading
        engine = LiveTradingEngine(config)
        
        print("🔄 Initializing AI components...")
        await engine.initialize()
        
        print("✅ All systems operational!")
        print("🎯 Starting live trading execution...")
        print("=" * 60)
        
        # Run for demo period (5 minutes)
        await asyncio.wait_for(
            engine.start_live_trading(),
            timeout=300  # 5 minutes
        )
        
    except asyncio.TimeoutError:
        print("\n⏰ Demo session completed (5 minutes)")
        await engine.stop_live_trading()
        
    except KeyboardInterrupt:
        print("\n⚠️  Manual stop requested...")
        await engine.stop_live_trading()
        
    except Exception as e:
        print(f"\n❌ Error during live trading: {e}")
        if 'engine' in locals():
            await engine.stop_live_trading()
    
    print("\n" + "=" * 60)
    print("✅ Live Trading Demo Complete")
    print("🎯 CroweTrade AI Infrastructure Ready for Production")
    print("🚀 Deploy to production: ./scripts/deploy-production.sh")
    print("=" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)
