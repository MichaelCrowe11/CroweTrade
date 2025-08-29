from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class FeatureDefinition:
    name: str
    category: str  # microstructure, vol, liquidity, nlp, cross_asset
    horizon: str  # tick, 1s, 10s, 1m, 5m, 1h, 1d
    dependencies: list[str] = field(default_factory=list)
    compute_fn: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    min_history: int = 0  # minimum ticks/bars needed
    is_stateless: bool = False


@dataclass
class FeatureSet:
    instrument: str
    timestamp: datetime
    features: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_vector(self, feature_names: list[str]) -> np.ndarray:
        return np.array([self.features.get(f, np.nan) for f in feature_names])
    
    def is_complete(self, required_features: list[str]) -> bool:
        return all(
            f in self.features and not np.isnan(self.features[f])
            for f in required_features
        )


class FeatureComputer(ABC):
    def __init__(self, definition: FeatureDefinition):
        self.definition = definition
        self.state = {}
    
    @abstractmethod
    def compute(self, data: Any) -> float:
        pass
    
    @abstractmethod
    def reset(self) -> None:
        pass
    
    def is_ready(self) -> bool:
        return True


class MicrostructureFeatures:
    @staticmethod
    def bid_ask_spread(order_book) -> float:
        if order_book.bids and order_book.asks:
            return order_book.asks[0][0] - order_book.bids[0][0]
        return np.nan
    
    @staticmethod
    def mid_price(order_book) -> float:
        if order_book.bids and order_book.asks:
            return (order_book.bids[0][0] + order_book.asks[0][0]) / 2
        return np.nan
    
    @staticmethod
    def book_imbalance(order_book, levels: int = 5) -> float:
        bid_vol = sum(b[1] for b in order_book.bids[:levels])
        ask_vol = sum(a[1] for a in order_book.asks[:levels])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0
    
    @staticmethod
    def effective_spread(trade_price: float, mid_price: float, trade_size: float) -> float:
        return 2 * abs(trade_price - mid_price)
    
    @staticmethod
    def price_impact(trade_price: float, pre_trade_mid: float, trade_size: float) -> float:
        return abs(trade_price - pre_trade_mid) / pre_trade_mid


class VolatilityFeatures:
    @staticmethod
    def realized_volatility(returns: np.ndarray, annualize: bool = True) -> float:
        if len(returns) < 2:
            return np.nan
        vol = np.std(returns)
        if annualize:
            vol *= np.sqrt(252 * 390 * 60)  # assuming minute bars
        return vol
    
    @staticmethod
    def garch_vol(returns: np.ndarray, omega: float = 0.00001,
                  alpha: float = 0.1, beta: float = 0.85) -> float:
        if len(returns) < 2:
            return np.nan
        
        var = np.var(returns)
        for r in returns:
            var = omega + alpha * r**2 + beta * var
        return np.sqrt(var)
    
    @staticmethod
    def yang_zhang_vol(open_: np.ndarray, high: np.ndarray, 
                      low: np.ndarray, close: np.ndarray) -> float:
        if len(close) < 2:
            return np.nan
        
        n = len(close)
        log_ho = np.log(high / open_)
        log_hc = np.log(high / close)
        log_lc = np.log(low / close)
        log_lo = np.log(low / open_)
        
        rs = log_ho * log_hc + log_lo * log_lc
        sigma_rs = np.sqrt(np.sum(rs) / n)
        
        log_cc = np.log(close[1:] / close[:-1])
        sigma_c = np.std(log_cc)
        
        log_oc = np.log(open_[1:] / close[:-1])
        sigma_o = np.std(log_oc)
        
        k = 0.34 / (1 + (n + 1) / (n - 1))
        sigma = np.sqrt(sigma_o**2 + k * sigma_c**2 + (1 - k) * sigma_rs**2)
        
        return sigma * np.sqrt(252)  # annualized