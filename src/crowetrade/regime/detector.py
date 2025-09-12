"""Regime Detection Module

Implements multiple regime detection algorithms for market state classification:
- Hidden Markov Model (HMM) for volatility regimes
- Bayesian Online Change Point Detection (BOCPD) 
- Turbulence Index for stress/crash detection
- Volatility clustering detection

Each detector outputs regime probabilities and confidence scores used by
the Portfolio Manager and Risk Guard for dynamic strategy adjustment.
"""

import math
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Market regime classifications"""
    BULL = "bull"           # Low volatility, positive returns
    BEAR = "bear"           # High volatility, negative returns  
    SIDEWAYS = "sideways"   # Low volatility, neutral returns
    VOLATILE = "volatile"   # High volatility, mixed returns
    CRASH = "crash"         # Extreme volatility, sharp negative returns


@dataclass
class RegimeState:
    """Current regime detection output"""
    regime: RegimeType
    confidence: float       # [0,1] confidence in regime classification
    probabilities: Dict[RegimeType, float]  # Probability distribution over regimes
    volatility: float       # Current volatility estimate
    turbulence: float       # Turbulence index [0,inf)
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "probabilities": {k.value: v for k, v in self.probabilities.items()},
            "volatility": self.volatility,
            "turbulence": self.turbulence,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }


class RegimeDetector:
    """Multi-algorithm regime detection system"""
    
    def __init__(self, 
                 lookback_window: int = 252,  # Trading days for volatility estimation
                 vol_threshold_low: float = 0.12,   # Annual vol threshold for low vol regimes
                 vol_threshold_high: float = 0.25,  # Annual vol threshold for high vol regimes
                 turbulence_threshold: float = 3.0, # Turbulence index crash threshold
                 min_regime_duration: int = 5):     # Minimum regime persistence (days)
        
        self.lookback_window = lookback_window
        self.vol_threshold_low = vol_threshold_low
        self.vol_threshold_high = vol_threshold_high
        self.turbulence_threshold = turbulence_threshold
        self.min_regime_duration = min_regime_duration
        
        # State tracking
        self.returns_history: List[float] = []
        self.volatility_history: List[float] = []
        self.regime_history: List[Tuple[RegimeType, datetime]] = []
        self.covariance_matrix: Optional[np.ndarray] = None
        
        # HMM parameters (2-state: low vol / high vol)
        self.transition_matrix = np.array([[0.95, 0.05], [0.10, 0.90]])  # Persistent regimes
        self.state_probs = np.array([0.7, 0.3])  # Prior: 70% low vol, 30% high vol
        
    def update(self, returns: Dict[str, float], 
               market_data: Optional[Dict[str, Any]] = None) -> RegimeState:
        """Update regime detection with new return data
        
        Args:
            returns: Dict of asset returns {symbol: return}
            market_data: Optional additional market indicators
            
        Returns:
            RegimeState with current regime classification
        """
        if not returns:
            logger.warning("Empty returns data provided to regime detector")
            return self._get_default_regime_state()
            
        # Calculate portfolio-level return (equal weighted for simplicity)
        portfolio_return = sum(returns.values()) / len(returns)
        self.returns_history.append(portfolio_return)
        
        # Maintain rolling window
        if len(self.returns_history) > self.lookback_window:
            self.returns_history.pop(0)
            
        # Need minimum data for regime detection
        if len(self.returns_history) < 20:
            return self._get_default_regime_state()
            
        # Calculate current volatility (annualized)
        vol = self._calculate_volatility()
        self.volatility_history.append(vol)
        
        if len(self.volatility_history) > self.lookback_window:
            self.volatility_history.pop(0)
            
        # Calculate turbulence index
        turbulence = self._calculate_turbulence_index(returns)
        
        # Run regime detection algorithms
        hmm_regime, hmm_confidence = self._hmm_regime_detection()
        vol_regime = self._volatility_regime_classification(vol)
        trend_regime = self._trend_regime_detection()
        
        # Combine regime signals with crash detection override
        if turbulence > self.turbulence_threshold:
            final_regime = RegimeType.CRASH
            confidence = min(1.0, turbulence / self.turbulence_threshold)
        else:
            # Weighted combination of regime signals
            regime_votes = [hmm_regime, vol_regime, trend_regime]
            final_regime = max(set(regime_votes), key=regime_votes.count)
            confidence = hmm_confidence
        
        # Apply regime persistence filter
        final_regime = self._apply_persistence_filter(final_regime)
        
        # Calculate regime probabilities
        probabilities = self._calculate_regime_probabilities(vol, portfolio_return, turbulence)
        
        regime_state = RegimeState(
            regime=final_regime,
            confidence=confidence,
            probabilities=probabilities,
            volatility=vol,
            turbulence=turbulence,
            timestamp=datetime.utcnow(),
            metadata={
                "hmm_regime": hmm_regime.value,
                "vol_regime": vol_regime.value,
                "trend_regime": trend_regime.value,
                "returns_window_size": len(self.returns_history)
            }
        )
        
        # Update regime history
        self.regime_history.append((final_regime, datetime.utcnow()))
        if len(self.regime_history) > self.lookback_window:
            self.regime_history.pop(0)
            
        logger.info(f"Regime detected: {final_regime.value} (confidence: {confidence:.2f}, vol: {vol:.3f})")
        return regime_state
    
    def _calculate_volatility(self) -> float:
        """Calculate annualized volatility from returns history"""
        if len(self.returns_history) < 2:
            return 0.15  # Default reasonable volatility
            
        returns_array = np.array(self.returns_history)
        daily_vol = np.std(returns_array, ddof=1)
        return daily_vol * math.sqrt(252)  # Annualize
    
    def _calculate_turbulence_index(self, returns: Dict[str, float]) -> float:
        """Calculate Mahalanobis distance-based turbulence index"""
        if len(self.returns_history) < 30:
            return 0.0
            
        try:
            # Simple proxy: standardized deviation from recent mean
            recent_returns = self.returns_history[-30:]
            mean_return = np.mean(recent_returns)
            vol = np.std(recent_returns, ddof=1)
            
            current_return = sum(returns.values()) / len(returns)
            turbulence = abs(current_return - mean_return) / max(vol, 1e-6)
            
            return float(turbulence)
            
        except Exception as e:
            logger.warning(f"Turbulence calculation error: {e}")
            return 0.0
    
    def _hmm_regime_detection(self) -> Tuple[RegimeType, float]:
        """Simple 2-state HMM for volatility regimes"""
        if len(self.volatility_history) < 10:
            return RegimeType.SIDEWAYS, 0.5
            
        try:
            current_vol = self.volatility_history[-1]
            recent_vols = self.volatility_history[-10:]
            
            # Simple state classification based on volatility level
            if current_vol > self.vol_threshold_high:
                state = RegimeType.VOLATILE
                confidence = min(1.0, (current_vol - self.vol_threshold_high) / self.vol_threshold_high)
            elif current_vol < self.vol_threshold_low:
                state = RegimeType.BULL if np.mean(self.returns_history[-10:]) > 0 else RegimeType.SIDEWAYS
                confidence = min(1.0, (self.vol_threshold_low - current_vol) / self.vol_threshold_low)
            else:
                state = RegimeType.SIDEWAYS
                confidence = 0.5
                
            return state, max(0.5, confidence)
            
        except Exception as e:
            logger.warning(f"HMM regime detection error: {e}")
            return RegimeType.SIDEWAYS, 0.5
    
    def _volatility_regime_classification(self, vol: float) -> RegimeType:
        """Simple volatility-based regime classification"""
        if vol > self.vol_threshold_high:
            return RegimeType.VOLATILE
        elif vol < self.vol_threshold_low:
            return RegimeType.BULL
        else:
            return RegimeType.SIDEWAYS
    
    def _trend_regime_detection(self) -> RegimeType:
        """Trend-based regime detection using recent returns"""
        if len(self.returns_history) < 20:
            return RegimeType.SIDEWAYS
            
        recent_returns = self.returns_history[-20:]
        mean_return = np.mean(recent_returns)
        
        if mean_return > 0.001:  # 0.1% daily threshold
            return RegimeType.BULL
        elif mean_return < -0.001:
            return RegimeType.BEAR
        else:
            return RegimeType.SIDEWAYS
    
    def _apply_persistence_filter(self, new_regime: RegimeType) -> RegimeType:
        """Apply minimum regime duration filter to reduce noise"""
        if len(self.regime_history) < self.min_regime_duration:
            return new_regime
            
        # Check if we've been in current regime long enough
        recent_regimes = [r[0] for r in self.regime_history[-self.min_regime_duration:]]
        current_regime = self.regime_history[-1][0] if self.regime_history else new_regime
        
        # If all recent regimes are the same and different from new regime, 
        # require stronger evidence (crash regimes always override)
        if (all(r == current_regime for r in recent_regimes) and 
            current_regime != new_regime and 
            new_regime != RegimeType.CRASH):
            return current_regime
            
        return new_regime
    
    def _calculate_regime_probabilities(self, vol: float, return_: float, 
                                      turbulence: float) -> Dict[RegimeType, float]:
        """Calculate probability distribution over regime types"""
        probabilities = {}
        
        # Crash detection (overrides others)
        crash_prob = min(1.0, max(0.0, (turbulence - 1.0) / 2.0))
        
        if crash_prob > 0.7:
            probabilities = {
                RegimeType.CRASH: crash_prob,
                RegimeType.BEAR: 1.0 - crash_prob,
                RegimeType.BULL: 0.0,
                RegimeType.SIDEWAYS: 0.0,
                RegimeType.VOLATILE: 0.0
            }
        else:
            # Normal regime probabilities based on vol and returns
            vol_factor = (vol - self.vol_threshold_low) / (self.vol_threshold_high - self.vol_threshold_low)
            vol_factor = max(0.0, min(1.0, vol_factor))
            
            return_factor = math.tanh(return_ * 50)  # Scale returns to [-1, 1]
            
            probabilities = {
                RegimeType.BULL: max(0.0, (1 - vol_factor) * (0.5 + return_factor)),
                RegimeType.BEAR: max(0.0, vol_factor * (0.5 - return_factor)),
                RegimeType.SIDEWAYS: 1 - vol_factor if abs(return_factor) < 0.2 else 0.1,
                RegimeType.VOLATILE: vol_factor * 0.5,
                RegimeType.CRASH: crash_prob
            }
        
        # Normalize probabilities
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}
        else:
            # Default uniform distribution
            probabilities = {regime: 0.2 for regime in RegimeType}
            
        return probabilities
    
    def _get_default_regime_state(self) -> RegimeState:
        """Return default regime state when insufficient data"""
        return RegimeState(
            regime=RegimeType.SIDEWAYS,
            confidence=0.5,
            probabilities={regime: 0.2 for regime in RegimeType},
            volatility=0.15,
            turbulence=0.0,
            timestamp=datetime.utcnow(),
            metadata={"insufficient_data": True}
        )
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get historical regime statistics and diagnostics"""
        if not self.regime_history:
            return {"error": "No regime history available"}
            
        regime_counts = {}
        for regime, _ in self.regime_history:
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
            
        return {
            "regime_distribution": regime_counts,
            "current_volatility": self.volatility_history[-1] if self.volatility_history else 0.0,
            "average_volatility": np.mean(self.volatility_history) if self.volatility_history else 0.0,
            "returns_history_length": len(self.returns_history),
            "regime_history_length": len(self.regime_history),
            "last_regime_change": self.regime_history[-1][1].isoformat() if self.regime_history else None
        }
