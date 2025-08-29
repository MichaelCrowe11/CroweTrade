from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime

from ..core.agent import AgentConfig, BaseAgent
from ..core.events import ExecutionReport
from ..core.events import RiskState as RiskStateEvent
from ..core.types import Symbol
from .manager import RiskManager


@dataclass
class RiskConfig(AgentConfig):
    initial_capital: float = 1000000.0
    var_limit: float = 0.02
    drawdown_limit: float = 0.05
    exposure_limit: float = 2.0
    check_interval: float = 1.0
    kill_switch_enabled: bool = True


class RiskGuard(BaseAgent):
    def __init__(self, config: RiskConfig):
        super().__init__(config)
        self.config: RiskConfig = config
        self.manager = RiskManager(config.initial_capital)
        self._kill_switch_active = False
        self._running = False
    
    async def on_start(self):
        self._running = True
        self.subscribe(ExecutionReport, self._on_execution)
        asyncio.create_task(self._monitor_risk())
    
    async def on_stop(self):
        self._running = False
    
    async def _on_execution(self, event: ExecutionReport):
        if event.status in ("FILLED", "PARTIAL"):
            self.manager.update_position(
                event.symbol,
                event.filled_qty,
                event.avg_price
            )
    
    async def _monitor_risk(self):
        while self._running:
            passed, violations = self.manager.check_limits(
                var_limit=self.config.var_limit,
                drawdown_limit=self.config.drawdown_limit,
                exposure_limit=self.config.exposure_limit
            )
            
            if not passed and self.config.kill_switch_enabled:
                await self._activate_kill_switch(violations)
            
            state_event = RiskStateEvent(
                event_id=f"risk_state_{datetime.utcnow().isoformat()}",
                pnl=self.manager.state.metrics.pnl,
                exposure=self.manager.state.metrics.exposure,
                var_estimate=self.manager.state.metrics.var_95,
                max_drawdown=self.manager.state.metrics.max_drawdown,
                positions=dict(self.manager.state.positions),
                risk_budget_remaining=1.0 - (
                    self.manager.state.metrics.current_drawdown / 
                    (self.config.drawdown_limit * self.config.initial_capital)
                )
            )
            
            await self.emit(state_event)
            await asyncio.sleep(self.config.check_interval)
    
    async def _activate_kill_switch(self, violations: list[str]):
        if self._kill_switch_active:
            return
        
        self._kill_switch_active = True
        
        for violation in violations:
            pass
    
    def pre_trade_check(
        self,
        symbol: Symbol,
        qty: float,
        price: float
    ) -> tuple[bool, str | None]:
        if self._kill_switch_active:
            return False, "Kill switch active"
        
        temp_manager = RiskManager(self.config.initial_capital)
        temp_manager.state = self.manager.state
        
        temp_manager.update_position(symbol, qty, price)
        
        passed, violations = temp_manager.check_limits(
            var_limit=self.config.var_limit,
            drawdown_limit=self.config.drawdown_limit,
            exposure_limit=self.config.exposure_limit
        )
        
        if not passed:
            return False, "; ".join(violations)
        
        return True, None