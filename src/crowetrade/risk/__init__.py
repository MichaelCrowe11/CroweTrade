from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

import numpy as np


class RiskMetric(Enum):
    VAR = "var"
    CVAR = "cvar"
    EXPECTED_SHORTFALL = "es"
    MAX_DRAWDOWN = "max_dd"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    EXPOSURE = "exposure"
    BETA = "beta"
    CORRELATION = "correlation"


class RiskLimit(Enum):
    POSITION = "position"
    NOTIONAL = "notional"
    VAR = "var"
    DRAWDOWN = "drawdown"
    EXPOSURE = "exposure"
    CONCENTRATION = "concentration"
    TURNOVER = "turnover"
    LEVERAGE = "leverage"


@dataclass
class RiskBudget:
    portfolio: str
    strategy: str
    limits: dict[RiskLimit, float]
    usage: dict[RiskLimit, float] = field(default_factory=dict)
    last_updated: datetime | None = None
    
    def is_within_limits(self) -> bool:
        for limit_type, limit_value in self.limits.items():
            if limit_type in self.usage and self.usage[limit_type] > limit_value:
                return False
        return True
    
    def get_utilization(self, limit_type: RiskLimit) -> float:
        if limit_type not in self.limits:
            return 0.0
        if limit_type not in self.usage:
            return 0.0
        return self.usage[limit_type] / self.limits[limit_type]
    
    def get_available(self, limit_type: RiskLimit) -> float:
        if limit_type not in self.limits:
            return float('inf')
        used = self.usage.get(limit_type, 0.0)
        return max(0, self.limits[limit_type] - used)


@dataclass
class RiskMetrics:
    timestamp: datetime
    portfolio: str
    metrics: dict[RiskMetric, float]
    positions: dict[str, float] = field(default_factory=dict)
    exposures: dict[str, float] = field(default_factory=dict)
    
    def get_total_exposure(self) -> float:
        return sum(abs(v) for v in self.exposures.values())
    
    def get_net_exposure(self) -> float:
        return sum(self.exposures.values())
    
    def get_gross_exposure(self) -> float:
        return sum(abs(v) for v in self.exposures.values())


class RiskCalculator:
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95,
                     method: str = "historical") -> float:
        if len(returns) == 0:
            return np.nan
        
        if method == "historical":
            return -np.percentile(returns, (1 - confidence) * 100)
        elif method == "parametric":
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = 1.645 if confidence == 0.95 else 2.326  # simplified
            return -(mean - z_score * std)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        if len(returns) == 0:
            return np.nan
        
        var = RiskCalculator.calculate_var(returns, confidence)
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) == 0:
            return var
        
        return -np.mean(tail_returns)
    
    @staticmethod
    def calculate_max_drawdown(values: np.ndarray) -> tuple[float, int, int]:
        if len(values) < 2:
            return 0.0, 0, 0
        
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax
        
        max_dd = np.min(drawdown)
        end_idx = np.argmin(drawdown)
        
        start_idx = 0
        for i in range(end_idx, -1, -1):
            if values[i] == cummax[end_idx]:
                start_idx = i
                break
        
        return abs(max_dd), start_idx, end_idx
    
    @staticmethod
    def calculate_sharpe(returns: np.ndarray, risk_free: float = 0.02,
                        periods: int = 252) -> float:
        if len(returns) < 2:
            return np.nan
        
        excess_returns = returns - risk_free / periods
        
        if np.std(excess_returns) == 0:
            return np.nan
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods)
    
    @staticmethod
    def calculate_sortino(returns: np.ndarray, target_return: float = 0.0,
                         periods: int = 252) -> float:
        if len(returns) < 2:
            return np.nan
        
        excess_returns = returns - target_return / periods
        downside_returns = np.minimum(excess_returns, 0)
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return np.nan
        
        return np.mean(excess_returns) / downside_std * np.sqrt(periods)
    
    @staticmethod
    def calculate_calmar(returns: np.ndarray, values: np.ndarray,
                        periods: int = 252) -> float:
        if len(returns) < 2 or len(values) < 2:
            return np.nan
        
        annual_return = np.mean(returns) * periods
        max_dd, _, _ = RiskCalculator.calculate_max_drawdown(values)
        
        if max_dd == 0:
            return np.nan
        
        return annual_return / max_dd


class ComplianceChecker(Protocol):
    def check_pre_trade(self, order: Any) -> tuple[bool, str | None]:
        ...
    
    def check_post_trade(self, fill: Any) -> tuple[bool, str | None]:
        ...
    
    def check_position_limits(self, positions: dict[str, float]) -> tuple[bool, str | None]:
        ...