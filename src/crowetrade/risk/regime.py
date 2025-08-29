from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass
class RegimeState:
    vol: float
    regime: str  # "low", "mid", "high"
    confidence: float = 0.0


class RegimeService:
    """Lightweight rolling-volatility regime detector (HMM/BOCPD placeholder)."""
    
    def __init__(self, window: int = 120, hi: float = 0.02, lo: float = 0.005):
        self.window = window
        self.hi = hi
        self.lo = lo
        self._rets: deque[float] = deque(maxlen=window)

    def update(self, ret: float) -> RegimeState:
        self._rets.append(ret)
        if len(self._rets) < 2:
            return RegimeState(0.0, "unknown", 0.0)
        
        mu = sum(self._rets) / len(self._rets)
        var = sum((r - mu) ** 2 for r in self._rets) / max(1, len(self._rets) - 1)
        vol = math.sqrt(var)
        
        if vol >= self.hi:
            regime = "high"
        elif vol <= self.lo:
            regime = "low"
        else:
            regime = "mid"
        
        confidence = min(len(self._rets) / self.window, 1.0)
        return RegimeState(vol=vol, regime=regime, confidence=confidence)


@dataclass
class RiskResult:
    ok: bool
    reason: str = "OK"
    metrics: dict = None


class RiskMetrics:
    """Calculate risk metrics for portfolio."""
    
    def __init__(self, dd_limit: float = -0.02, var_k: float = 2.0):
        self.dd_limit = dd_limit
        self.var_k = var_k  # multiplier for std proxy
        self.pnl_peak = 0.0
        self.current_dd = 0.0

    def check_drawdown(self, pnl: float) -> RiskResult:
        """Check if current drawdown exceeds limit."""
        self.pnl_peak = max(self.pnl_peak, pnl)
        self.current_dd = (pnl - self.pnl_peak) / max(abs(self.pnl_peak), 1.0)
        
        if self.current_dd <= self.dd_limit:
            return RiskResult(False, f"DRAWDOWN_BREACH: {self.current_dd:.2%}")
        return RiskResult(True)

    def var_proxy(self, rets: Iterable[float]) -> float:
        """Calculate simple VaR proxy using standard deviation."""
        xs = list(rets)
        if not xs:
            return 0.0
        mu = sum(xs) / len(xs)
        var = sum((x - mu) ** 2 for x in xs) / max(1, len(xs) - 1)
        return math.sqrt(var) * self.var_k
    
    def expected_shortfall(self, rets: Iterable[float], alpha: float = 0.05) -> float:
        """Calculate Expected Shortfall (CVaR) at given confidence level."""
        sorted_rets = sorted(rets)
        if not sorted_rets:
            return 0.0
        
        cutoff_idx = max(1, int(len(sorted_rets) * alpha))
        tail_losses = sorted_rets[:cutoff_idx]
        return sum(tail_losses) / len(tail_losses) if tail_losses else 0.0
    
    def turbulence_index(self, returns: list[float], baseline_cov) -> float:
        """Mahalanobis distance-based turbulence metric."""
        if len(returns) < 2:
            return 0.0
        
        mu = sum(returns) / len(returns)
        centered = [r - mu for r in returns]
        
        # Simplified for single asset
        var = sum(c ** 2 for c in centered) / max(1, len(centered) - 1)
        if var > 0:
            return abs(centered[-1]) / math.sqrt(var)
        return 0.0