"""Regime-aware trading agent that adapts to market conditions"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np

from crowetrade.core.agent import BaseAgent
from crowetrade.core.events import Event, EventType
from crowetrade.ml.regime_detector import (
    RegimeDetector,
    MarketRegime,
    RegimeDetectionResult
)
from crowetrade.data.feature_store import PersistentFeatureStore

logger = logging.getLogger(__name__)


class RegimeAwareAgent(BaseAgent):
    """
    Agent that detects market regimes and adapts trading strategy accordingly
    """
    
    def __init__(
        self,
        agent_id: str,
        regime_detector: Optional[RegimeDetector] = None,
        feature_store: Optional[PersistentFeatureStore] = None,
        adapt_position_sizing: bool = True,
        adapt_risk_limits: bool = True,
        regime_change_cooldown: int = 300  # seconds
    ):
        super().__init__(agent_id)
        
        self.regime_detector = regime_detector or RegimeDetector()
        self.feature_store = feature_store or PersistentFeatureStore()
        
        # Configuration
        self.adapt_position_sizing = adapt_position_sizing
        self.adapt_risk_limits = adapt_risk_limits
        self.regime_change_cooldown = regime_change_cooldown
        
        # State tracking
        self.current_regime: Optional[MarketRegime] = None
        self.last_regime_change: float = 0
        self.regime_detection_results: List[RegimeDetectionResult] = []
        
        # Buffered data for regime detection
        self.returns_buffer: List[float] = []
        self.volumes_buffer: List[float] = []
        self.max_buffer_size = 500
        
        # Performance tracking by regime
        self.regime_performance: Dict[MarketRegime, Dict[str, float]] = {
            regime: {'trades': 0, 'pnl': 0, 'win_rate': 0}
            for regime in MarketRegime
        }
    
    async def process_event(self, event: Event) -> Optional[Event]:
        """Process events and detect regime changes"""
        
        if event.type == EventType.MARKET_DATA:
            return await self._handle_market_data(event)
        
        elif event.type == EventType.SIGNAL:
            return await self._handle_signal_with_regime(event)
        
        elif event.type == EventType.POSITION_UPDATE:
            return await self._update_performance_tracking(event)
        
        return None
    
    async def _handle_market_data(self, event: Event) -> Optional[Event]:
        """Process market data and detect regime"""
        
        market_data = event.data
        
        # Update buffers
        if 'returns' in market_data:
            self.returns_buffer.extend(market_data['returns'])
            if len(self.returns_buffer) > self.max_buffer_size:
                self.returns_buffer = self.returns_buffer[-self.max_buffer_size:]
        
        if 'volumes' in market_data:
            self.volumes_buffer.extend(market_data['volumes'])
            if len(self.volumes_buffer) > self.max_buffer_size:
                self.volumes_buffer = self.volumes_buffer[-self.max_buffer_size:]
        
        # Need sufficient data for regime detection
        if len(self.returns_buffer) < self.regime_detector.lookback_window:
            return None
        
        # Detect regime
        returns_array = np.array(self.returns_buffer)
        volumes_array = np.array(self.volumes_buffer) if self.volumes_buffer else None
        
        # Get VIX if available
        vix = None
        vix_feature = self.feature_store.get('vix_level')
        if vix_feature:
            vix = vix_feature.data.get('value')
        
        # Detect current regime
        detection_result = self.regime_detector.detect_regime(
            returns_array,
            volumes_array,
            vix=vix
        )
        
        self.regime_detection_results.append(detection_result)
        
        # Store in feature store
        self.feature_store.put(
            f'regime_{self.agent_id}',
            {
                'regime': detection_result.current_regime.value,
                'probability': detection_result.regime_probability,
                'scores': {k.value: v for k, v in detection_result.regime_scores.items()},
                'transition_probability': detection_result.transition_probability,
                'timestamp': datetime.now().isoformat()
            },
            ttl_seconds=3600
        )
        
        # Check for regime change
        if self._should_change_regime(detection_result):
            return await self._handle_regime_change(detection_result)
        
        return None
    
    async def _handle_signal_with_regime(self, event: Event) -> Optional[Event]:
        """Adjust signals based on current regime"""
        
        if self.current_regime is None:
            return event  # Pass through if no regime detected
        
        signal_data = event.data.copy()
        
        # Get regime recommendations
        recommendations = self.regime_detector.get_regime_recommendations(
            self.current_regime
        )
        
        # Adapt position sizing
        if self.adapt_position_sizing:
            position_multiplier = recommendations['position_sizing']
            
            if 'position_sizes' in signal_data:
                for symbol in signal_data['position_sizes']:
                    signal_data['position_sizes'][symbol] *= position_multiplier
            
            signal_data['regime_position_adjustment'] = position_multiplier
        
        # Adapt risk limits
        if self.adapt_risk_limits:
            risk_level = recommendations['risk_level']
            
            # Map risk level to multipliers
            risk_multipliers = {
                'very_low': 0.3,
                'low': 0.5,
                'low-medium': 0.7,
                'medium': 1.0,
                'medium-high': 1.2,
                'high': 1.5
            }
            
            risk_mult = risk_multipliers.get(risk_level, 1.0)
            signal_data['risk_multiplier'] = risk_mult
        
        # Add regime context
        signal_data['regime'] = self.current_regime.value
        signal_data['regime_strategies'] = recommendations['strategies']
        signal_data['regime_allocation'] = recommendations['asset_allocation']
        
        # Filter signals based on regime strategies
        if 'strategies' in signal_data:
            regime_strategies = set(recommendations['strategies'])
            signal_strategies = set(signal_data['strategies'])
            
            # Only keep signals aligned with regime
            aligned_strategies = signal_strategies & regime_strategies
            if aligned_strategies:
                signal_data['strategies'] = list(aligned_strategies)
            else:
                # No aligned strategies, reduce signal strength
                if 'confidence' in signal_data:
                    for symbol in signal_data['confidence']:
                        signal_data['confidence'][symbol] *= 0.5
        
        # Create adjusted event
        return Event(
            type=EventType.SIGNAL,
            source=self.agent_id,
            data=signal_data
        )
    
    def _should_change_regime(
        self,
        detection_result: RegimeDetectionResult
    ) -> bool:
        """Check if regime should be changed"""
        
        # No current regime
        if self.current_regime is None:
            return True
        
        # Check if regime actually changed
        if detection_result.current_regime == self.current_regime:
            return False
        
        # Check cooldown period
        time_since_change = datetime.now().timestamp() - self.last_regime_change
        if time_since_change < self.regime_change_cooldown:
            return False
        
        # Check confidence threshold
        if detection_result.regime_probability < 0.6:
            return False
        
        # High transition probability
        if detection_result.transition_probability > 0.7:
            return True
        
        # Sustained regime change (check last few detections)
        if len(self.regime_detection_results) >= 3:
            last_regimes = [r.current_regime for r in self.regime_detection_results[-3:]]
            if all(r == detection_result.current_regime for r in last_regimes):
                return True
        
        return False
    
    async def _handle_regime_change(
        self,
        detection_result: RegimeDetectionResult
    ) -> Event:
        """Handle regime change event"""
        
        old_regime = self.current_regime
        self.current_regime = detection_result.current_regime
        self.last_regime_change = datetime.now().timestamp()
        
        logger.info(
            f"Regime change: {old_regime.value if old_regime else 'None'} -> "
            f"{self.current_regime.value} (confidence: {detection_result.regime_probability:.2%})"
        )
        
        # Get recommendations for new regime
        recommendations = self.regime_detector.get_regime_recommendations(
            self.current_regime
        )
        
        # Create regime change event
        return Event(
            type=EventType.REGIME_CHANGE,
            source=self.agent_id,
            data={
                'old_regime': old_regime.value if old_regime else None,
                'new_regime': self.current_regime.value,
                'probability': detection_result.regime_probability,
                'transition_probability': detection_result.transition_probability,
                'recommendations': recommendations,
                'features': {
                    'returns_mean': detection_result.features.returns_mean,
                    'returns_std': detection_result.features.returns_std,
                    'trend_strength': detection_result.features.trend_strength,
                    'volatility_ratio': detection_result.features.volatility_ratio
                },
                'timestamp': datetime.now().isoformat()
            }
        )
    
    async def _update_performance_tracking(self, event: Event) -> None:
        """Track performance by regime"""
        
        if self.current_regime is None:
            return
        
        position_data = event.data
        
        if 'realized_pnl' in position_data:
            pnl = position_data['realized_pnl']
            self.regime_performance[self.current_regime]['pnl'] += pnl
            self.regime_performance[self.current_regime]['trades'] += 1
            
            # Update win rate
            if pnl > 0:
                trades = self.regime_performance[self.current_regime]['trades']
                wins = self.regime_performance[self.current_regime].get('wins', 0) + 1
                self.regime_performance[self.current_regime]['wins'] = wins
                self.regime_performance[self.current_regime]['win_rate'] = wins / trades
    
    def train_on_historical(
        self,
        historical_returns: np.ndarray,
        historical_volumes: Optional[np.ndarray] = None
    ) -> None:
        """Train regime detector on historical data"""
        
        logger.info(f"Training regime detector on {len(historical_returns)} samples")
        self.regime_detector.train_gmm(historical_returns, historical_volumes)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics"""
        
        metrics = {
            'agent_id': self.agent_id,
            'current_regime': self.current_regime.value if self.current_regime else None,
            'regime_detections': len(self.regime_detection_results),
            'buffer_size': len(self.returns_buffer)
        }
        
        # Add performance by regime
        for regime, perf in self.regime_performance.items():
            if perf['trades'] > 0:
                metrics[f'{regime.value}_pnl'] = perf['pnl']
                metrics[f'{regime.value}_trades'] = perf['trades']
                metrics[f'{regime.value}_win_rate'] = perf['win_rate']
        
        # Latest detection results
        if self.regime_detection_results:
            latest = self.regime_detection_results[-1]
            metrics['latest_regime_probability'] = latest.regime_probability
            metrics['latest_transition_probability'] = latest.transition_probability
        
        return metrics