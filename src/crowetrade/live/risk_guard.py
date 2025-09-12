from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PnLMode(str, Enum):
    """Mode describing how a PnL value should be interpreted."""

    INCREMENTAL = "incremental"  # value is a delta to add
    ABSOLUTE = "absolute"  # value is the new absolute cumulative PnL


@dataclass
class PnLUpdate:
    """Structured PnL update for explicit semantics.

    Attributes
    ----------
    value : float
        Either a delta (incremental) or an absolute total depending on `mode`.
    mode : PnLMode
        Interpretation of the value.
    source : Optional[str]
        Optional tag / provenance (e.g. 'backtest', 'live_fill', 'reprice').
    """

    value: float
    mode: PnLMode = PnLMode.INCREMENTAL
    source: Optional[str] = None


@dataclass
class PnLState:
    """Container tracking cumulative PnL and drawdowns.

    High water mark (HWM) is the maximum observed cumulative PnL under *any* mode.
    Drawdown is maintained as HWM - cumulative (never negative). Limits are applied
    to the absolute drawdown distance (not percentage) consistent with legacy tests.
    """

    cumulative: float = 0.0
    high_water_mark: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0

    def _post_update(self) -> None:
        if self.cumulative > self.high_water_mark:
            self.high_water_mark = self.cumulative
        self.current_drawdown = max(0.0, self.high_water_mark - self.cumulative)
        if self.current_drawdown > self.max_drawdown:
            self.max_drawdown = self.current_drawdown

    def apply_increment(self, delta: float) -> None:
        self.cumulative += float(delta)
        self._post_update()

    def set_total(self, total: float) -> None:
        self.cumulative = float(total)
        self._post_update()


class RiskGuard:
    """RiskGuard with explicit PnL semantics.

    Supports both incremental and absolute PnL updates via `PnLUpdate`.
    Existing code using `update_pnl(delta)` retains *incremental* meaning.
    Tests relying on snapshot semantics should call `set_total_pnl` or
    `apply_update(PnLUpdate(value, PnLMode.ABSOLUTE))` for clarity.
    """

    def __init__(self, dd_limit: float, var_limit: float):
        self.dd_limit = float(dd_limit)
        self.var_limit = float(var_limit)
        self.state = PnLState()
        self.kill_switch_active = False

    # ------------------------------------------------------------------
    # Backwards compatible API (incremental by default)
    # ------------------------------------------------------------------
    def update_pnl(self, value: float) -> None:  # pragma: no cover - legacy name
        self.update_incremental(float(value))

    def update_incremental(self, delta: float) -> None:
        self.state.apply_increment(delta)
        self._post_risk_checks()

    def set_total_pnl(self, total: float) -> None:
        self.state.set_total(total)
        self._post_risk_checks()

    # ------------------------------------------------------------------
    # New explicit API
    # ------------------------------------------------------------------
    def apply_update(self, upd: PnLUpdate) -> None:
        if upd.mode == PnLMode.ABSOLUTE:
            self.state.set_total(upd.value)
        else:  # default incremental
            self.state.apply_increment(upd.value)
        self._post_risk_checks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _post_risk_checks(self) -> None:
        # Cap drawdown at HWM (already true) and evaluate kill switch
        if self.state.current_drawdown > self.state.high_water_mark:
            self.state.current_drawdown = self.state.high_water_mark
        self.kill_switch_active = self.state.current_drawdown > self.dd_limit

    @property
    def hwm(self) -> float:
        return self.state.high_water_mark

    @property
    def current_dd(self) -> float:
        return self.state.current_drawdown

    @current_dd.setter
    def current_dd(self, value: float) -> None:
        self.state.current_drawdown = max(0.0, float(value))
        if self.state.current_drawdown > self.dd_limit:
            self.kill_switch_active = True

    @property
    def current_drawdown(self) -> float:
        return self.state.current_drawdown

    @property
    def max_drawdown(self) -> float:
        return self.state.max_drawdown

    def pretrade_check(self, exposure: float, var_est: float) -> bool:
        if self.kill_switch_active:
            return False
        if self.state.max_drawdown > self.dd_limit:
            return False
        if float(var_est) > self.var_limit:
            return False
        return True

    def reset_kill_switch(self) -> None:
        if self.state.current_drawdown < self.dd_limit * 0.5:
            self.kill_switch_active = False
