"""Integrated risk management system combining all risk components."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from crowetrade.config.settings import RiskConfig
from crowetrade.core.types import OrderId
from crowetrade.risk.circuit_breaker import CircuitBreaker, BreakerConfig, TripReason
from crowetrade.risk.guard import RiskGuard
from crowetrade.risk.manager import RiskManager

logger = logging.getLogger(__name__)


class RiskAction(Enum):
    """Risk management actions."""
    
    ALLOW = "allow"
    REJECT = "reject"
    REDUCE = "reduce"
    FLATTEN = "flatten"
    HALT = "halt"


@dataclass
class RiskDecision:
    """Risk management decision."""
    
    action: RiskAction
    reason: str
    details: Dict[str, Any]
    suggested_size: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class IntegratedRiskSystem:
    """Integrated risk management system.
    
    Combines multiple risk components:
    - Circuit breakers for catastrophic protection
    - Risk guards for pre-trade checks
    - Risk manager for metrics and limits
    - Position limits and margin management
    """
    
    def __init__(self, config: RiskConfig):
        """Initialize integrated risk system.
        
        Args:
            config: Risk configuration
        """
        self.config = config
        
        # Initialize components
        self.circuit_breaker = CircuitBreaker(
            BreakerConfig(
                max_drawdown=config.max_drawdown,
                intraday_loss_limit=config.intraday_loss_limit,
                max_volatility_spike=3.0,
                max_position_concentration=config.max_position_size,
                max_gross_exposure=config.max_leverage,
            )
        )
        
        self.risk_guard = RiskGuard(
            dd_limit=config.max_drawdown,
            var_limit=config.var_limit,
        )
        
        self.risk_manager = RiskManager()
        
        # Position tracking
        self.positions: Dict[str, float] = {}
        self.gross_exposure = 0.0
        self.net_exposure = 0.0
        
        # Margin tracking
        self.initial_margin = 0.0
        self.maintenance_margin = 0.0
        self.available_margin = 100000.0  # Starting capital
        
        # Risk state
        self.is_reducing_only = False
        self.is_halted = False
        
        # Setup callbacks
        self._setup_callbacks()
    
    def _setup_callbacks(self) -> None:
        """Setup circuit breaker callbacks."""
        
        def on_breaker_trip(reason: TripReason, details: Dict[str, Any]):
            logger.critical(f"RISK SYSTEM: Circuit breaker tripped - {reason.value}")
            self.is_halted = True
            
            # Take immediate action based on reason
            if reason in [TripReason.DRAWDOWN_LIMIT, TripReason.LOSS_VELOCITY]:
                # Severe loss - flatten all positions
                asyncio.create_task(self._flatten_all_positions())
            elif reason == TripReason.POSITION_LIMIT:
                # Position too large - go to reduce only
                self.is_reducing_only = True
        
        def on_breaker_reset():
            logger.info("RISK SYSTEM: Circuit breaker reset")
            self.is_halted = False
            self.is_reducing_only = False
        
        self.circuit_breaker.add_trip_callback(on_breaker_trip)
        self.circuit_breaker.add_reset_callback(on_breaker_reset)
    
    async def check_pre_trade(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> RiskDecision:
        """Perform comprehensive pre-trade risk check.
        
        Args:
            symbol: Symbol to trade
            side: "buy" or "sell"
            quantity: Order quantity
            price: Expected price
            current_positions: Current position sizes
            
        Returns:
            Risk decision with action to take
        """
        # Check if system is halted
        if self.is_halted:
            return RiskDecision(
                action=RiskAction.HALT,
                reason="System halted by circuit breaker",
                details={"breaker_state": self.circuit_breaker.state.value},
            )
        
        # Check circuit breaker
        if not self.circuit_breaker.check():
            return RiskDecision(
                action=RiskAction.REJECT,
                reason="Circuit breaker open",
                details=self.circuit_breaker.get_status(),
            )
        
        # Update positions if provided
        if current_positions:
            self.positions = current_positions.copy()
        
        # Calculate new position after trade
        current_position = self.positions.get(symbol, 0.0)
        if side == "buy":
            new_position = current_position + quantity
        else:
            new_position = current_position - quantity
        
        # Check if reduce-only mode
        if self.is_reducing_only:
            if abs(new_position) > abs(current_position):
                return RiskDecision(
                    action=RiskAction.REJECT,
                    reason="System in reduce-only mode",
                    details={"current": current_position, "proposed": new_position},
                )
        
        # Check restricted symbols
        if symbol in self.config.restricted_symbols:
            return RiskDecision(
                action=RiskAction.REJECT,
                reason="Symbol is restricted",
                details={"symbol": symbol},
            )
        
        # Check position limits
        position_value = abs(new_position * price)
        portfolio_value = self.available_margin  # Simplified
        
        # Symbol-specific limit
        symbol_limit = self.config.position_limits.get(
            symbol,
            self.config.max_position_size
        )
        if position_value > portfolio_value * symbol_limit:
            # Suggest reduced size
            max_value = portfolio_value * symbol_limit
            suggested_quantity = max_value / price
            
            if suggested_quantity < quantity * 0.1:  # Too small
                return RiskDecision(
                    action=RiskAction.REJECT,
                    reason="Position limit exceeded",
                    details={
                        "limit": symbol_limit,
                        "requested_value": position_value,
                        "max_value": max_value,
                    },
                )
            else:
                return RiskDecision(
                    action=RiskAction.REDUCE,
                    reason="Position size reduced to fit limits",
                    details={
                        "original_quantity": quantity,
                        "suggested_quantity": suggested_quantity,
                    },
                    suggested_size=suggested_quantity,
                )
        
        # Calculate new exposures
        new_positions = self.positions.copy()
        new_positions[symbol] = new_position
        
        gross_exposure = sum(abs(pos * price) for pos in new_positions.values())
        net_exposure = sum(pos * price for pos in new_positions.values())
        
        # Check leverage limits
        leverage = gross_exposure / portfolio_value if portfolio_value > 0 else 0
        if leverage > self.config.max_leverage:
            return RiskDecision(
                action=RiskAction.REJECT,
                reason="Leverage limit exceeded",
                details={
                    "max_leverage": self.config.max_leverage,
                    "would_be_leverage": leverage,
                },
            )
        
        # Check margin requirements
        required_margin = self._calculate_margin_requirement(
            new_positions,
            price
        )
        if required_margin > self.available_margin:
            return RiskDecision(
                action=RiskAction.REJECT,
                reason="Insufficient margin",
                details={
                    "required": required_margin,
                    "available": self.available_margin,
                },
            )
        
        # Perform traditional risk checks
        var_estimate = abs(quantity * price * 0.02)  # Simplified VaR
        if not self.risk_guard.pretrade_check(gross_exposure, var_estimate):
            return RiskDecision(
                action=RiskAction.REJECT,
                reason="Risk guard check failed",
                details={
                    "exposure": gross_exposure,
                    "var": var_estimate,
                    "drawdown": self.risk_guard.current_drawdown,
                },
            )
        
        # Check concentration
        position_concentration = position_value / portfolio_value if portfolio_value > 0 else 0
        if position_concentration > 0.15:  # Warning threshold
            logger.warning(
                f"High concentration in {symbol}: {position_concentration:.1%}"
            )
        
        # All checks passed
        return RiskDecision(
            action=RiskAction.ALLOW,
            reason="All risk checks passed",
            details={
                "leverage": leverage,
                "concentration": position_concentration,
                "margin_used": required_margin / self.available_margin,
            },
        )
    
    def update_pnl(self, pnl: float) -> None:
        """Update P&L across all risk components.
        
        Args:
            pnl: P&L update
        """
        # Update risk guard
        self.risk_guard.update_pnl(pnl)
        
        # Update risk manager
        self.risk_manager.update_pnl(symbol="PORTFOLIO", pnl=pnl)
        
        # Update circuit breaker
        self.circuit_breaker.update_metrics(
            drawdown=self.risk_guard.current_drawdown,
            daily_pnl=self.risk_manager.get_daily_pnl(),
        )
        
        # Update available margin
        self.available_margin += pnl
    
    def update_positions(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
    ) -> None:
        """Update position tracking.
        
        Args:
            positions: Current positions by symbol
            prices: Current prices by symbol
        """
        self.positions = positions.copy()
        
        # Calculate exposures
        self.gross_exposure = sum(
            abs(pos * prices.get(symbol, 0))
            for symbol, pos in positions.items()
        )
        
        self.net_exposure = sum(
            pos * prices.get(symbol, 0)
            for symbol, pos in positions.items()
        )
        
        # Calculate concentration
        portfolio_value = self.available_margin
        if portfolio_value > 0 and positions:
            max_position_value = max(
                abs(pos * prices.get(symbol, 0))
                for symbol, pos in positions.items()
            )
            concentration = max_position_value / portfolio_value
        else:
            concentration = 0.0
        
        # Update circuit breaker
        self.circuit_breaker.update_metrics(
            position_concentration=concentration,
            gross_exposure=self.gross_exposure / portfolio_value if portfolio_value > 0 else 0,
        )
    
    def _calculate_margin_requirement(
        self,
        positions: Dict[str, float],
        reference_price: float,
    ) -> float:
        """Calculate margin requirement for positions.
        
        Args:
            positions: Position sizes by symbol
            reference_price: Reference price for calculation
            
        Returns:
            Required margin
        """
        # Simplified margin calculation
        # In production, would use symbol-specific requirements
        total_value = sum(abs(pos * reference_price) for pos in positions.values())
        
        # Assume 25% margin requirement for equities
        initial_margin = total_value * 0.25
        
        # Add buffer for volatility
        volatility_buffer = total_value * 0.05
        
        return initial_margin + volatility_buffer
    
    async def _flatten_all_positions(self) -> None:
        """Emergency flatten all positions."""
        logger.critical("EMERGENCY: Flattening all positions")
        
        # This would connect to execution system to close all positions
        # For now, just log
        for symbol, position in self.positions.items():
            if position != 0:
                logger.info(f"Would flatten {symbol}: {position}")
        
        # Clear positions
        self.positions.clear()
        self.gross_exposure = 0.0
        self.net_exposure = 0.0
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get comprehensive risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        portfolio_value = self.available_margin
        
        return {
            "circuit_breaker": self.circuit_breaker.get_status(),
            "drawdown": self.risk_guard.current_drawdown,
            "max_drawdown": self.risk_guard.max_drawdown,
            "daily_pnl": self.risk_manager.get_daily_pnl(),
            "sharpe_ratio": self.risk_manager.get_sharpe_ratio(),
            "gross_exposure": self.gross_exposure,
            "net_exposure": self.net_exposure,
            "leverage": self.gross_exposure / portfolio_value if portfolio_value > 0 else 0,
            "margin_used": 1.0 - (self.available_margin / portfolio_value) if portfolio_value > 0 else 0,
            "positions": len(self.positions),
            "is_halted": self.is_halted,
            "is_reducing_only": self.is_reducing_only,
        }
    
    def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Trigger emergency stop.
        
        Args:
            reason: Reason for emergency stop
        """
        logger.critical(f"EMERGENCY STOP: {reason}")
        self.circuit_breaker.manual_trip(reason)
        self.is_halted = True
        asyncio.create_task(self._flatten_all_positions())