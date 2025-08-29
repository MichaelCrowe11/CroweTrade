"""Position Sizing Module - Stateless Position Calculation"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class SizingMethod(Enum):
    FIXED = "fixed"
    KELLY = "kelly"
    RISK_PARITY = "risk_parity"
    VOLATILITY_SCALED = "volatility_scaled"


@dataclass
class SizingConfig:
    method: SizingMethod
    max_position_size: float
    max_portfolio_exposure: float
    risk_per_trade: float
    params: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "max_position_size": self.max_position_size,
            "max_portfolio_exposure": self.max_portfolio_exposure,
            "risk_per_trade": self.risk_per_trade,
            "params": self.params
        }


@dataclass
class PositionSize:
    symbol: str
    quantity: float
    notional_value: float
    risk_amount: float
    sizing_method: str
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "notional_value": self.notional_value,
            "risk_amount": self.risk_amount,
            "sizing_method": self.sizing_method,
            "metadata": self.metadata
        }


class PositionSizer:
    def __init__(self, config: SizingConfig):
        self.config = config
        self.sizing_functions = {
            SizingMethod.FIXED: self._fixed_sizing,
            SizingMethod.KELLY: self._kelly_sizing,
            SizingMethod.RISK_PARITY: self._risk_parity_sizing,
            SizingMethod.VOLATILITY_SCALED: self._volatility_scaled_sizing
        }
        
    def calculate_size(self, signal: dict[str, Any], portfolio_state: dict[str, Any], 
                      market_data: dict[str, Any]) -> PositionSize:
        sizing_fn = self.sizing_functions.get(self.config.method, self._fixed_sizing)
        
        quantity = sizing_fn(signal, portfolio_state, market_data)
        
        quantity = self._apply_constraints(quantity, signal, portfolio_state)
        
        price = market_data.get("price", 0.0)
        notional_value = quantity * price
        
        volatility = market_data.get("volatility", 0.0)
        risk_amount = notional_value * volatility
        
        return PositionSize(
            symbol=signal.get("symbol"),
            quantity=quantity,
            notional_value=notional_value,
            risk_amount=risk_amount,
            sizing_method=self.config.method.value,
            metadata={
                "signal_strength": signal.get("strength"),
                "portfolio_nav": portfolio_state.get("nav"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def _fixed_sizing(self, signal: dict[str, Any], portfolio_state: dict[str, Any], 
                     market_data: dict[str, Any]) -> float:
        nav = portfolio_state.get("nav", 0.0)
        price = market_data.get("price", 1.0)
        
        position_value = nav * self.config.risk_per_trade
        return position_value / price
    
    def _kelly_sizing(self, signal: dict[str, Any], portfolio_state: dict[str, Any], 
                     market_data: dict[str, Any]) -> float:
        win_prob = signal.get("win_probability", 0.5)
        win_loss_ratio = signal.get("win_loss_ratio", 1.0)
        
        kelly_fraction = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        kelly_fraction = max(0, min(kelly_fraction, 0.25))
        
        nav = portfolio_state.get("nav", 0.0)
        price = market_data.get("price", 1.0)
        
        position_value = nav * kelly_fraction
        return position_value / price
    
    def _risk_parity_sizing(self, signal: dict[str, Any], portfolio_state: dict[str, Any], 
                           market_data: dict[str, Any]) -> float:
        volatility = market_data.get("volatility", 0.01)
        target_risk = self.config.risk_per_trade
        
        nav = portfolio_state.get("nav", 0.0)
        price = market_data.get("price", 1.0)
        
        position_value = (target_risk * nav) / volatility
        return position_value / price
    
    def _volatility_scaled_sizing(self, signal: dict[str, Any], portfolio_state: dict[str, Any], 
                                 market_data: dict[str, Any]) -> float:
        volatility = market_data.get("volatility", 0.01)
        target_volatility = self.config.params.get("target_volatility", 0.02)
        
        scale = min(1.0, target_volatility / volatility)
        
        nav = portfolio_state.get("nav", 0.0)
        price = market_data.get("price", 1.0)
        
        base_position_value = nav * self.config.risk_per_trade
        return (base_position_value * scale) / price
    
    def _apply_constraints(self, quantity: float, signal: dict[str, Any], 
                          portfolio_state: dict[str, Any]) -> float:
        nav = portfolio_state.get("nav", 0.0)
        current_exposure = portfolio_state.get("total_exposure", 0.0)
        
        max_quantity = (nav * self.config.max_position_size) / signal.get("price", 1.0)
        quantity = min(quantity, max_quantity)
        
        new_exposure = current_exposure + (quantity * signal.get("price", 1.0))
        max_allowed_exposure = nav * self.config.max_portfolio_exposure
        
        if new_exposure > max_allowed_exposure:
            allowed_additional = max_allowed_exposure - current_exposure
            quantity = allowed_additional / signal.get("price", 1.0)
        
        return max(0.0, quantity)