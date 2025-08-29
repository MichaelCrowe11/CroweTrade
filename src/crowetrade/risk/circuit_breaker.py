"""Circuit breaker implementation for risk management.

Provides multiple layers of protection including kill switches,
rate limits, and anomaly detection.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class BreakerState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Breaker tripped, blocking all operations
    HALF_OPEN = "half_open"  # Testing if system recovered


class TripReason(Enum):
    """Reasons for circuit breaker trips."""
    
    DRAWDOWN_LIMIT = "drawdown_limit"
    LOSS_VELOCITY = "loss_velocity"
    ERROR_RATE = "error_rate"
    LATENCY = "latency"
    DATA_QUALITY = "data_quality"
    MANUAL = "manual"
    VOLATILITY_SPIKE = "volatility_spike"
    POSITION_LIMIT = "position_limit"
    MARGIN_CALL = "margin_call"


@dataclass
class BreakerConfig:
    """Circuit breaker configuration."""
    
    # Drawdown limits
    max_drawdown: float = 0.05  # 5% max drawdown
    intraday_loss_limit: float = 0.02  # 2% daily loss limit
    
    # Loss velocity (speed of losses)
    loss_velocity_window: int = 60  # seconds
    max_loss_velocity: float = 0.01  # 1% per minute max loss rate
    
    # Error thresholds
    error_rate_window: int = 300  # 5 minutes
    max_error_rate: float = 0.1  # 10% error rate threshold
    max_consecutive_errors: int = 5
    
    # Latency thresholds
    max_latency_ms: int = 1000  # 1 second max latency
    latency_check_window: int = 60  # seconds
    
    # Recovery settings
    recovery_time: int = 300  # 5 minutes before attempting recovery
    test_trades_in_recovery: int = 3  # Number of test trades in half-open
    
    # Volatility limits
    max_volatility_spike: float = 3.0  # 3x normal volatility
    
    # Position limits
    max_position_concentration: float = 0.2  # 20% in single position
    max_gross_exposure: float = 1.5  # 150% gross exposure


@dataclass
class BreakerEvent:
    """Circuit breaker event record."""
    
    timestamp: datetime
    state: BreakerState
    reason: Optional[TripReason]
    details: Dict[str, Any]
    recovery_time: Optional[datetime] = None


class CircuitBreaker:
    """Main circuit breaker implementation."""
    
    def __init__(self, config: BreakerConfig = None):
        """Initialize circuit breaker.
        
        Args:
            config: Configuration settings
        """
        self.config = config or BreakerConfig()
        self.state = BreakerState.CLOSED
        self.trip_reason: Optional[TripReason] = None
        self.trip_time: Optional[datetime] = None
        self.recovery_time: Optional[datetime] = None
        
        # Tracking metrics
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.error_count = 0
        self.consecutive_errors = 0
        self.recent_latencies: List[float] = []
        self.test_trade_results: List[bool] = []
        
        # Event history
        self.events: List[BreakerEvent] = []
        
        # Callbacks
        self.trip_callbacks: List[Callable] = []
        self.reset_callbacks: List[Callable] = []
        
        # Loss tracking for velocity
        self.loss_history: List[Tuple[datetime, float]] = []
    
    def check(self) -> bool:
        """Check if operations are allowed.
        
        Returns:
            True if operations allowed, False if breaker is open
        """
        if self.state == BreakerState.OPEN:
            # Check if recovery time has passed
            if self.recovery_time and datetime.utcnow() >= self.recovery_time:
                self._transition_to_half_open()
            else:
                return False
        
        return self.state != BreakerState.OPEN
    
    def update_metrics(
        self,
        drawdown: Optional[float] = None,
        daily_pnl: Optional[float] = None,
        latency_ms: Optional[float] = None,
        error: Optional[bool] = None,
        volatility: Optional[float] = None,
        position_concentration: Optional[float] = None,
        gross_exposure: Optional[float] = None,
    ) -> None:
        """Update tracked metrics and check for trips.
        
        Args:
            drawdown: Current drawdown
            daily_pnl: Daily P&L
            latency_ms: Latest operation latency
            error: Whether latest operation errored
            volatility: Current market volatility
            position_concentration: Max position concentration
            gross_exposure: Total gross exposure
        """
        # Update drawdown
        if drawdown is not None:
            self.current_drawdown = drawdown
            if drawdown > self.config.max_drawdown:
                self._trip(TripReason.DRAWDOWN_LIMIT, {"drawdown": drawdown})
        
        # Update daily P&L
        if daily_pnl is not None:
            self.daily_pnl = daily_pnl
            if daily_pnl < -self.config.intraday_loss_limit:
                self._trip(TripReason.DRAWDOWN_LIMIT, {"daily_loss": daily_pnl})
            
            # Track loss velocity
            self._update_loss_velocity(daily_pnl)
        
        # Update latency
        if latency_ms is not None:
            self.recent_latencies.append(latency_ms)
            # Keep only recent latencies
            cutoff_time = datetime.utcnow() - timedelta(
                seconds=self.config.latency_check_window
            )
            self.recent_latencies = self.recent_latencies[-100:]  # Keep last 100
            
            if latency_ms > self.config.max_latency_ms:
                avg_latency = sum(self.recent_latencies) / len(self.recent_latencies)
                if avg_latency > self.config.max_latency_ms:
                    self._trip(TripReason.LATENCY, {"latency_ms": latency_ms})
        
        # Update error tracking
        if error is not None:
            if error:
                self.error_count += 1
                self.consecutive_errors += 1
                
                if self.consecutive_errors >= self.config.max_consecutive_errors:
                    self._trip(
                        TripReason.ERROR_RATE,
                        {"consecutive_errors": self.consecutive_errors}
                    )
            else:
                self.consecutive_errors = 0
        
        # Check volatility spike
        if volatility is not None:
            if volatility > self.config.max_volatility_spike:
                self._trip(
                    TripReason.VOLATILITY_SPIKE,
                    {"volatility": volatility}
                )
        
        # Check position limits
        if position_concentration is not None:
            if position_concentration > self.config.max_position_concentration:
                self._trip(
                    TripReason.POSITION_LIMIT,
                    {"concentration": position_concentration}
                )
        
        if gross_exposure is not None:
            if gross_exposure > self.config.max_gross_exposure:
                self._trip(
                    TripReason.POSITION_LIMIT,
                    {"gross_exposure": gross_exposure}
                )
    
    def _update_loss_velocity(self, pnl: float) -> None:
        """Update loss velocity tracking.
        
        Args:
            pnl: Current P&L
        """
        now = datetime.utcnow()
        self.loss_history.append((now, pnl))
        
        # Remove old entries
        cutoff = now - timedelta(seconds=self.config.loss_velocity_window)
        self.loss_history = [
            (t, p) for t, p in self.loss_history if t > cutoff
        ]
        
        # Calculate velocity if we have enough data
        if len(self.loss_history) >= 2:
            time_span = (
                self.loss_history[-1][0] - self.loss_history[0][0]
            ).total_seconds()
            
            if time_span > 0:
                pnl_change = self.loss_history[-1][1] - self.loss_history[0][1]
                velocity = pnl_change / (time_span / 60)  # Per minute
                
                if velocity < -self.config.max_loss_velocity:
                    self._trip(
                        TripReason.LOSS_VELOCITY,
                        {"velocity": velocity, "window_seconds": time_span}
                    )
    
    def _trip(self, reason: TripReason, details: Dict[str, Any]) -> None:
        """Trip the circuit breaker.
        
        Args:
            reason: Reason for trip
            details: Additional details
        """
        if self.state == BreakerState.OPEN:
            return  # Already tripped
        
        self.state = BreakerState.OPEN
        self.trip_reason = reason
        self.trip_time = datetime.utcnow()
        self.recovery_time = self.trip_time + timedelta(
            seconds=self.config.recovery_time
        )
        
        # Log event
        event = BreakerEvent(
            timestamp=self.trip_time,
            state=BreakerState.OPEN,
            reason=reason,
            details=details,
            recovery_time=self.recovery_time,
        )
        self.events.append(event)
        
        logger.warning(
            f"Circuit breaker TRIPPED: {reason.value} - {details}"
        )
        
        # Execute callbacks
        for callback in self.trip_callbacks:
            try:
                callback(reason, details)
            except Exception as e:
                logger.error(f"Error in trip callback: {e}")
    
    def _transition_to_half_open(self) -> None:
        """Transition to half-open state for testing."""
        self.state = BreakerState.HALF_OPEN
        self.test_trade_results = []
        
        logger.info("Circuit breaker transitioning to HALF-OPEN for testing")
        
        event = BreakerEvent(
            timestamp=datetime.utcnow(),
            state=BreakerState.HALF_OPEN,
            reason=None,
            details={"previous_reason": self.trip_reason.value if self.trip_reason else None},
        )
        self.events.append(event)
    
    def record_test_trade(self, success: bool) -> None:
        """Record result of test trade in half-open state.
        
        Args:
            success: Whether test trade succeeded
        """
        if self.state != BreakerState.HALF_OPEN:
            return
        
        self.test_trade_results.append(success)
        
        if not success:
            # Failed test, go back to open
            self.state = BreakerState.OPEN
            self.recovery_time = datetime.utcnow() + timedelta(
                seconds=self.config.recovery_time
            )
            logger.warning("Test trade failed, circuit breaker returning to OPEN")
        
        elif len(self.test_trade_results) >= self.config.test_trades_in_recovery:
            # Enough successful tests, close breaker
            if all(self.test_trade_results[-self.config.test_trades_in_recovery:]):
                self.reset()
    
    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.state = BreakerState.CLOSED
        self.trip_reason = None
        self.trip_time = None
        self.recovery_time = None
        self.consecutive_errors = 0
        self.test_trade_results = []
        
        logger.info("Circuit breaker RESET to CLOSED state")
        
        event = BreakerEvent(
            timestamp=datetime.utcnow(),
            state=BreakerState.CLOSED,
            reason=None,
            details={"reset": True},
        )
        self.events.append(event)
        
        # Execute callbacks
        for callback in self.reset_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in reset callback: {e}")
    
    def manual_trip(self, reason: str = "Manual intervention") -> None:
        """Manually trip the circuit breaker.
        
        Args:
            reason: Reason for manual trip
        """
        self._trip(TripReason.MANUAL, {"reason": reason})
    
    def add_trip_callback(self, callback: Callable) -> None:
        """Add callback for trip events.
        
        Args:
            callback: Function to call on trip
        """
        self.trip_callbacks.append(callback)
    
    def add_reset_callback(self, callback: Callable) -> None:
        """Add callback for reset events.
        
        Args:
            callback: Function to call on reset
        """
        self.reset_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current breaker status.
        
        Returns:
            Status dictionary
        """
        return {
            "state": self.state.value,
            "trip_reason": self.trip_reason.value if self.trip_reason else None,
            "trip_time": self.trip_time.isoformat() if self.trip_time else None,
            "recovery_time": self.recovery_time.isoformat() if self.recovery_time else None,
            "current_drawdown": self.current_drawdown,
            "daily_pnl": self.daily_pnl,
            "consecutive_errors": self.consecutive_errors,
            "recent_events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "state": e.state.value,
                    "reason": e.reason.value if e.reason else None,
                }
                for e in self.events[-10:]  # Last 10 events
            ],
        }


class MultiLayerBreaker:
    """Multi-layer circuit breaker system with different thresholds."""
    
    def __init__(self):
        """Initialize multi-layer breaker."""
        # Multiple breakers with different sensitivities
        self.breakers = {
            "aggressive": CircuitBreaker(
                BreakerConfig(
                    max_drawdown=0.03,  # 3% for aggressive
                    intraday_loss_limit=0.01,  # 1% daily
                    max_loss_velocity=0.005,  # 0.5% per minute
                )
            ),
            "normal": CircuitBreaker(
                BreakerConfig()  # Default settings
            ),
            "conservative": CircuitBreaker(
                BreakerConfig(
                    max_drawdown=0.10,  # 10% for conservative
                    intraday_loss_limit=0.05,  # 5% daily
                    max_loss_velocity=0.02,  # 2% per minute
                )
            ),
        }
        
        self.active_layer = "normal"
    
    def check(self, layer: Optional[str] = None) -> bool:
        """Check if operations allowed at specified layer.
        
        Args:
            layer: Specific layer to check, or active layer
            
        Returns:
            True if operations allowed
        """
        check_layer = layer or self.active_layer
        return self.breakers[check_layer].check()
    
    def update_all(self, **metrics) -> None:
        """Update all breaker layers with metrics.
        
        Args:
            **metrics: Metric values to update
        """
        for breaker in self.breakers.values():
            breaker.update_metrics(**metrics)
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all layers.
        
        Returns:
            Status by layer
        """
        return {
            name: breaker.get_status()
            for name, breaker in self.breakers.items()
        }