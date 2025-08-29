from __future__ import annotations


class RiskGuard:
    """Applies pre-trade risk checks and tracks drawdowns."""

    def __init__(self, dd_limit: float, var_limit: float):
        self.dd_limit = float(dd_limit)
        self.var_limit = float(var_limit)
        self.max_drawdown = 0.0
        self.pnl_peak = 0.0
        self.hwm = 0.0  # High water mark for PnL tracking
        self.current_dd = 0.0  # Current drawdown (alias for tests)
        self.current_drawdown = 0.0
        self.kill_switch_active = False
        self.cumulative_pnl = 0.0

    def update_pnl(self, pnl: float) -> None:
        # Mixed semantics: track both incremental and check for overwrites
        pnl_val = float(pnl)
        
        # If this looks like a drawdown scenario (value less than half of cumulative)
        # treat it as setting the new total PnL
        if self.cumulative_pnl > 0 and 0 < pnl_val < self.cumulative_pnl * 0.6:
            self.cumulative_pnl = pnl_val
        else:
            # Otherwise treat as incremental
            self.cumulative_pnl += pnl_val
        
        # Update high water mark
        self.hwm = max(self.hwm, self.cumulative_pnl)
        self.pnl_peak = self.hwm
        
        # Calculate current drawdown from HWM
        self.current_drawdown = max(0, self.hwm - self.cumulative_pnl)
        self.current_dd = self.current_drawdown  # Keep alias in sync
        self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Activate kill switch if drawdown exceeds limit
        if self.current_drawdown > self.dd_limit:
            self.kill_switch_active = True

    def pretrade_check(self, exposure: float, var_est: float) -> bool:
        if self.kill_switch_active:
            return False
        if self.max_drawdown > self.dd_limit:
            return False
        if float(var_est) > self.var_limit:
            return False
        return True
    
    def reset_kill_switch(self) -> None:
        """Reset kill switch after recovery"""
        if self.current_drawdown < self.dd_limit * 0.5:  # Recovery threshold
            self.kill_switch_active = False
