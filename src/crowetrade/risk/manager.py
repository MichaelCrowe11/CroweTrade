from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from ..core.types import Symbol


@dataclass
class RiskMetrics:
    pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    exposure: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0


@dataclass
class RiskState:
    positions: dict[Symbol, float] = field(default_factory=dict)
    prices: dict[Symbol, float] = field(default_factory=dict)
    costs: dict[Symbol, float] = field(default_factory=dict)
    metrics: RiskMetrics = field(default_factory=RiskMetrics)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    pnl_peak: float = 0.0
    pnl_history: list[float] = field(default_factory=list)
    return_history: list[float] = field(default_factory=list)


class RiskManager:
    def __init__(self, initial_capital: float = 1000000.0):
        self.initial_capital = initial_capital
        self.state = RiskState()
        self._volatility_window = 20
        self._var_confidence = 0.95
        self._es_confidence = 0.95
    
    def update_position(self, symbol: Symbol, qty: float, price: float):
        old_qty = self.state.positions.get(symbol, 0)
        self.state.positions[symbol] = qty
        
        if qty != 0:
            if old_qty == 0:
                self.state.costs[symbol] = price
            else:
                old_cost = self.state.costs.get(symbol, price)
                self.state.costs[symbol] = (
                    (old_cost * old_qty + price * (qty - old_qty)) / qty
                )
        
        self.state.prices[symbol] = price
        self._update_metrics()
    
    def update_price(self, symbol: Symbol, price: float):
        self.state.prices[symbol] = price
        self._update_metrics()
    
    def _update_metrics(self):
        metrics = self.state.metrics
        
        realized = 0.0
        unrealized = 0.0
        gross = 0.0
        net = 0.0
        
        for symbol, qty in self.state.positions.items():
            if qty == 0:
                continue
            
            price = self.state.prices.get(symbol, 0)
            cost = self.state.costs.get(symbol, price)
            
            position_value = qty * price
            position_cost = qty * cost
            position_pnl = position_value - position_cost
            
            unrealized += position_pnl
            gross += abs(position_value)
            net += position_value
        
        metrics.unrealized_pnl = unrealized
        metrics.realized_pnl = realized
        metrics.pnl = realized + unrealized
        metrics.gross_exposure = gross
        metrics.net_exposure = net
        metrics.exposure = gross
        
        self.state.pnl_history.append(metrics.pnl)
        if len(self.state.pnl_history) > 1:
            returns = np.diff(self.state.pnl_history) / self.initial_capital
            self.state.return_history = list(returns)
        
        self._update_risk_metrics()
        self._update_performance_metrics()
    
    def _update_risk_metrics(self):
        metrics = self.state.metrics
        
        self.state.pnl_peak = max(self.state.pnl_peak, metrics.pnl)
        metrics.current_drawdown = self.state.pnl_peak - metrics.pnl
        metrics.max_drawdown = max(metrics.max_drawdown, metrics.current_drawdown)
        
        if len(self.state.return_history) >= self._volatility_window:
            returns = np.array(self.state.return_history[-self._volatility_window:])
            
            if len(returns) > 0:
                sorted_returns = np.sort(returns)
                var_index = int((1 - self._var_confidence) * len(sorted_returns))
                metrics.var_95 = -sorted_returns[var_index] if var_index < len(sorted_returns) else 0
                
                var_index_99 = int((1 - 0.99) * len(sorted_returns))
                metrics.var_99 = -sorted_returns[var_index_99] if var_index_99 < len(sorted_returns) else 0
                
                es_returns = sorted_returns[:var_index] if var_index > 0 else sorted_returns[:1]
                metrics.expected_shortfall = -np.mean(es_returns) if len(es_returns) > 0 else 0
    
    def _update_performance_metrics(self):
        metrics = self.state.metrics
        
        if len(self.state.return_history) < 2:
            return
        
        returns = np.array(self.state.return_history)
        
        if len(returns) > 0:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return > 0:
                metrics.sharpe_ratio = (mean_return * 252) / (std_return * np.sqrt(252))
                
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns)
                    if downside_std > 0:
                        metrics.sortino_ratio = (mean_return * 252) / (downside_std * np.sqrt(252))
            
            if metrics.max_drawdown > 0:
                annual_return = mean_return * 252
                metrics.calmar_ratio = annual_return / (metrics.max_drawdown / self.initial_capital)
    
    def check_limits(
        self,
        var_limit: float = 0.02,
        drawdown_limit: float = 0.05,
        exposure_limit: float = 2.0
    ) -> tuple[bool, list[str]]:
        violations = []
        metrics = self.state.metrics
        
        if metrics.var_95 > var_limit:
            violations.append(f"VaR exceeded: {metrics.var_95:.4f} > {var_limit:.4f}")
        
        dd_pct = metrics.current_drawdown / self.initial_capital
        if dd_pct > drawdown_limit:
            violations.append(f"Drawdown exceeded: {dd_pct:.4f} > {drawdown_limit:.4f}")
        
        exposure_pct = metrics.gross_exposure / self.initial_capital
        if exposure_pct > exposure_limit:
            violations.append(f"Exposure exceeded: {exposure_pct:.4f} > {exposure_limit:.4f}")
        
        return len(violations) == 0, violations