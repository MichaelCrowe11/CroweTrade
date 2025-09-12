"""Regime Detection Agent

Event-driven agent that monitors market data and publishes regime change
notifications to the message bus. Integrates with Portfolio Manager and
Risk Guard for dynamic strategy adjustment based on market conditions.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from crowetrade.core.agent import BaseAgent, AgentConfig
from crowetrade.core.contracts import MarketDataEvent, RegimeChangeEvent
from crowetrade.regime.detector import RegimeDetector, RegimeType, RegimeState
from crowetrade.messaging.event_bus import EventBus

logger = logging.getLogger(__name__)


class RegimeDetectionAgent(BaseAgent):
    """Agent responsible for market regime detection and notification"""
    
    def __init__(self, 
                 config: AgentConfig,
                 event_bus: EventBus,
                 lookback_window: int = 252,
                 update_frequency: timedelta = timedelta(hours=1)):
        super().__init__(config)
        self.event_bus = event_bus
        self.update_frequency = update_frequency
        self.last_update = datetime.min
        self.last_regime = RegimeType.SIDEWAYS
        
        # Initialize regime detector
        self.detector = RegimeDetector(
            lookback_window=lookback_window,
            vol_threshold_low=config.policy.get("vol_threshold_low", 0.12),
            vol_threshold_high=config.policy.get("vol_threshold_high", 0.25),
            turbulence_threshold=config.policy.get("turbulence_threshold", 3.0),
            min_regime_duration=config.policy.get("min_regime_duration", 5)
        )
        
        # Market data buffer
        self.market_data_buffer: Dict[str, float] = {}
        self.last_market_update = datetime.min
        
        logger.info(f"RegimeDetectionAgent {config.agent_id} initialized with policy: {config.policy}")
    
    async def start(self) -> None:
        """Start the regime detection agent"""
        await super().start()
        
        # Subscribe to market data events
        await self.event_bus.subscribe("market_data", self._handle_market_data)
        await self.event_bus.subscribe("returns_calculated", self._handle_returns_data)
        
        # Start periodic regime detection
        asyncio.create_task(self._periodic_regime_check())
        
        logger.info(f"RegimeDetectionAgent {self.config.agent_id} started")
    
    async def stop(self) -> None:
        """Stop the regime detection agent"""
        await self.event_bus.unsubscribe("market_data", self._handle_market_data)
        await self.event_bus.unsubscribe("returns_calculated", self._handle_returns_data)
        await super().stop()
        logger.info(f"RegimeDetectionAgent {self.config.agent_id} stopped")
    
    async def _handle_market_data(self, event: MarketDataEvent) -> None:
        """Handle incoming market data for regime detection"""
        try:
            # Buffer market data for regime calculation
            if hasattr(event, 'symbol') and hasattr(event, 'price'):
                self.market_data_buffer[event.symbol] = event.price
                self.last_market_update = datetime.utcnow()
                
            # Trigger regime check if enough time has passed
            if (datetime.utcnow() - self.last_update) > self.update_frequency:
                await self._check_regime()
                
        except Exception as e:
            logger.error(f"Error handling market data in RegimeDetectionAgent: {e}")
            await self._handle_error(e)
    
    async def _handle_returns_data(self, event: Dict[str, Any]) -> None:
        """Handle calculated returns data for regime detection"""
        try:
            if 'returns' in event and isinstance(event['returns'], dict):
                returns_data = event['returns']
                
                # Update regime detection with new returns
                regime_state = self.detector.update(returns_data)
                
                # Check if regime has changed significantly
                if (regime_state.regime != self.last_regime or 
                    regime_state.confidence > 0.8):
                    await self._publish_regime_change(regime_state)
                    self.last_regime = regime_state.regime
                
                self.last_update = datetime.utcnow()
                
        except Exception as e:
            logger.error(f"Error handling returns data in RegimeDetectionAgent: {e}")
            await self._handle_error(e)
    
    async def _periodic_regime_check(self) -> None:
        """Periodic regime detection check"""
        while self._running:
            try:
                await asyncio.sleep(self.update_frequency.total_seconds())
                
                # Only check if we have recent market data
                if (datetime.utcnow() - self.last_market_update) < timedelta(minutes=30):
                    await self._check_regime()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic regime check: {e}")
                await self._handle_error(e)
    
    async def _check_regime(self) -> None:
        """Perform regime detection check"""
        try:
            if not self.market_data_buffer:
                logger.debug("No market data available for regime detection")
                return
            
            # Calculate simple returns from price buffer
            # In production, this would use proper returns calculation
            returns = {}
            for symbol, price in self.market_data_buffer.items():
                # Simplified return calculation - in practice, use proper price history
                returns[symbol] = 0.001  # Placeholder return
            
            # Update regime detection
            regime_state = self.detector.update(returns)
            
            # Publish regime update
            await self._publish_regime_state(regime_state)
            
            # Check for significant regime changes
            if regime_state.regime != self.last_regime:
                await self._publish_regime_change(regime_state)
                self.last_regime = regime_state.regime
                
        except Exception as e:
            logger.error(f"Error in regime check: {e}")
            await self._handle_error(e)
    
    async def _publish_regime_change(self, regime_state: RegimeState) -> None:
        """Publish regime change event to the event bus"""
        try:
            regime_change_event = RegimeChangeEvent(
                timestamp=regime_state.timestamp,
                previous_regime=self.last_regime.value,
                new_regime=regime_state.regime.value,
                confidence=regime_state.confidence,
                volatility=regime_state.volatility,
                turbulence=regime_state.turbulence,
                probabilities=regime_state.probabilities,
                metadata=regime_state.metadata or {}
            )
            
            await self.event_bus.publish("regime_change", regime_change_event)
            
            logger.info(f"Regime change detected: {self.last_regime.value} -> {regime_state.regime.value} "
                       f"(confidence: {regime_state.confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error publishing regime change event: {e}")
    
    async def _publish_regime_state(self, regime_state: RegimeState) -> None:
        """Publish current regime state for monitoring"""
        try:
            await self.event_bus.publish("regime_state", {
                "agent_id": self.config.agent_id,
                "regime": regime_state.regime.value,
                "confidence": regime_state.confidence,
                "volatility": regime_state.volatility,
                "turbulence": regime_state.turbulence,
                "timestamp": regime_state.timestamp.isoformat(),
                "probabilities": {k.value: v for k, v in regime_state.probabilities.items()}
            })
            
        except Exception as e:
            logger.error(f"Error publishing regime state: {e}")
    
    async def get_current_regime(self) -> Optional[RegimeState]:
        """Get the current regime state"""
        try:
            if not self.market_data_buffer:
                return None
                
            # Use cached data or trigger update
            returns = {symbol: 0.001 for symbol in self.market_data_buffer.keys()}
            return self.detector.update(returns)
            
        except Exception as e:
            logger.error(f"Error getting current regime: {e}")
            return None
    
    async def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime detection statistics"""
        try:
            stats = self.detector.get_regime_statistics()
            stats.update({
                "agent_id": self.config.agent_id,
                "last_update": self.last_update.isoformat() if self.last_update != datetime.min else None,
                "market_data_symbols": list(self.market_data_buffer.keys()),
                "update_frequency_minutes": self.update_frequency.total_seconds() / 60
            })
            return stats
            
        except Exception as e:
            logger.error(f"Error getting regime statistics: {e}")
            return {"error": str(e)}
    
    async def _handle_error(self, error: Exception) -> None:
        """Handle errors in regime detection"""
        error_event = {
            "agent_id": self.config.agent_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            await self.event_bus.publish("agent_error", error_event)
        except Exception as e:
            logger.error(f"Failed to publish error event: {e}")
        
        # Update agent health status
        self._health_status = "error"
        self._last_error = error
