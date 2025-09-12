from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PnLMode(Enum):
    """PnL update mode"""
    INCREMENTAL = "incremental"  # Value is a delta to add
    ABSOLUTE = "absolute"      # Value is a total snapshot


@dataclass
class PnLUpdate:
    """Structured PnL update with explicit mode"""
    value: float
    mode: PnLMode = PnLMode.INCREMENTAL
    timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PnLState:
    """Tracks cumulative and high-water mark P&L"""
    cumulative: float = 0.0
    high_water_mark: float = 0.0
    
    def update(self, update: PnLUpdate) -> None:
        """Update P&L state based on mode"""
        if update.mode == PnLMode.INCREMENTAL:
            self.cumulative += update.value
        else:  # ABSOLUTE
            self.cumulative = update.value
        
        self.high_water_mark = max(self.high_water_mark, self.cumulative)
    
    @property
    def drawdown(self) -> float:
        """Current drawdown from high water mark"""
        if self.high_water_mark <= 0:
            return 0.0
        # Cap drawdown at 1.0 (100%) for cases where cumulative goes negative
        dd = (self.high_water_mark - self.cumulative) / self.high_water_mark
        return min(1.0, max(0.0, dd))


class RiskGuard:
    """Risk management agent with explicit PnL semantics"""
    
    def __init__(
        self,
        max_position_size: float = 100000,
        max_portfolio_risk: float = 0.02,
        max_drawdown: float = 0.05,
        daily_loss_limit: float = 10000,
        position_limits: Optional[Dict[str, float]] = None,
        kill_switch_loss: float = 50000,
        recovery_threshold: float = 0.8
    ):
        # Limits
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit
        self.position_limits = position_limits or {}
        self.kill_switch_loss = kill_switch_loss
        self.recovery_threshold = recovery_threshold
        
        # State
        self.pnl_state = PnLState()
        self.daily_pnl = 0.0
        self.kill_switch_active = False
        self.positions: Dict[str, float] = {}
        
    def update_pnl(self, update: PnLUpdate) -> None:
        """Update P&L with explicit mode"""
        old_cumulative = self.pnl_state.cumulative
        self.pnl_state.update(update)
        
        # Track daily P&L (always incremental)
        if update.mode == PnLMode.INCREMENTAL:
            self.daily_pnl += update.value
        else:
            # For absolute updates, compute the delta
            delta = self.pnl_state.cumulative - old_cumulative
            self.daily_pnl += delta
        
        # Check for kill switch activation
        if self.pnl_state.cumulative <= -self.kill_switch_loss:
            if not self.kill_switch_active:
                logger.critical(f"Kill switch activated! Loss: {self.pnl_state.cumulative}")
                self.kill_switch_active = True
        
        # Check for recovery
        elif self.kill_switch_active and self.pnl_state.cumulative > -self.kill_switch_loss * self.recovery_threshold:
            logger.info(f"Kill switch deactivated. P&L recovered to: {self.pnl_state.cumulative}")
            self.kill_switch_active = False
    
    def update_pnl_incremental(self, delta: float) -> None:
        """Convenience method for incremental updates"""
        self.update_pnl(PnLUpdate(value=delta, mode=PnLMode.INCREMENTAL))
    
    def update_pnl_absolute(self, total: float) -> None:
        """Convenience method for absolute updates"""
        self.update_pnl(PnLUpdate(value=total, mode=PnLMode.ABSOLUTE))
    
    def pretrade_check(
        self,
        symbol: str,
        quantity: float,
        price: float,
        current_position: float = 0
    ) -> tuple[bool, str]:
        """Check if trade passes risk limits"""
        
        if self.kill_switch_active:
            return False, "Kill switch active"
        
        # Check drawdown
        if self.pnl_state.drawdown > self.max_drawdown:
            return False, f"Drawdown {self.pnl_state.drawdown:.2%} exceeds limit"
        
        # Check daily loss
        if self.daily_pnl <= -self.daily_loss_limit:
            return False, f"Daily loss ${self.daily_pnl:,.2f} exceeds limit"
        
        # Check position size
        position_value = abs(quantity * price)
        if position_value > self.max_position_size:
            return False, f"Position size ${position_value:,.2f} exceeds limit"
        
        # Check symbol-specific limits
        if symbol in self.position_limits:
            new_position = current_position + quantity
            if abs(new_position * price) > self.position_limits[symbol]:
                return False, f"Would exceed {symbol} position limit"
        
        return True, "Risk check passed"
    
    def reset_daily_pnl(self) -> None:
        """Reset daily P&L counter (call at start of trading day)"""
        self.daily_pnl = 0.0
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics"""
        return {
            "cumulative_pnl": self.pnl_state.cumulative,
            "high_water_mark": self.pnl_state.high_water_mark,
            "drawdown": self.pnl_state.drawdown,
            "daily_pnl": self.daily_pnl,
            "kill_switch_active": self.kill_switch_active,
            "position_count": len(self.positions),
            "total_exposure": sum(abs(v) for v in self.positions.values())
        }
    
    def update_position(self, symbol: str, quantity: float, price: float) -> None:
        """Update position tracking"""
        if symbol not in self.positions:
            self.positions[symbol] = 0
        self.positions[symbol] += quantity
        
        # Remove if flat
        if abs(self.positions[symbol]) < 1e-9:
            del self.positions[symbol]