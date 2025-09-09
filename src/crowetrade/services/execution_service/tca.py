"""TCA - Transaction Cost Analysis Service"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from crowetrade.core.contracts import Fill
from crowetrade.core.events import ChildOrder


@dataclass
class TCAReport:
    """Transaction cost analysis report"""
    instrument: str
    order_id: str
    arrival_price: float
    avg_fill_price: float
    total_qty: float
    spread_cost_bps: float
    impact_cost_bps: float
    timing_cost_bps: float
    total_cost_bps: float
    timestamp: datetime


class TransactionCostAnalysis:
    """TCA engine for post-trade analysis"""
    
    def __init__(self):
        self.reports: list[TCAReport] = []
        
    def analyze_execution(
        self, 
        order: ChildOrder, 
        fills: list[Fill], 
        market_data: dict[str, Any] | None = None
    ) -> TCAReport:
        """Analyze execution quality"""
        
        if not fills:
            # No fills to analyze
            return TCAReport(
                instrument=order.instrument,
                order_id=order.parent_id or "unknown",
                arrival_price=0.0,
                avg_fill_price=0.0,
                total_qty=0.0,
                spread_cost_bps=0.0,
                impact_cost_bps=0.0,
                timing_cost_bps=0.0,
                total_cost_bps=0.0,
                timestamp=datetime.utcnow()
            )
            
        # Calculate volume-weighted average fill price
        total_notional = sum(fill.qty * fill.price for fill in fills)
        total_qty = sum(fill.qty for fill in fills)
        avg_fill_price = total_notional / total_qty if total_qty > 0 else 0.0
        
    # Arrival price: prefer provided market_data, fallback to avg fill
        arrival_price = market_data.get("arrival_price", avg_fill_price) if market_data else avg_fill_price
        
        # Calculate cost components (simplified)
        spread_cost_bps = self._calculate_spread_cost(order, fills, market_data)
        impact_cost_bps = self._calculate_impact_cost(order, fills, market_data)
        timing_cost_bps = self._calculate_timing_cost(order, fills, market_data)
        
        total_cost_bps = spread_cost_bps + impact_cost_bps + timing_cost_bps
        
        report = TCAReport(
            instrument=order.instrument,
            order_id=order.parent_id or "unknown",
            arrival_price=arrival_price,
            avg_fill_price=avg_fill_price,
            total_qty=total_qty,
            spread_cost_bps=spread_cost_bps,
            impact_cost_bps=impact_cost_bps,
            timing_cost_bps=timing_cost_bps,
            total_cost_bps=total_cost_bps,
            timestamp=datetime.utcnow()
        )
        
        self.reports.append(report)
        return report
        
    def _calculate_spread_cost(
        self, 
        order: ChildOrder, 
        fills: list[Fill], 
        market_data: dict[str, Any] | None
    ) -> float:
        """Calculate spread crossing cost in bps"""
        if market_data and "spread" in market_data:
            # Use actual spread if available
            spread_pct = market_data["spread"]
            return spread_pct * 10000 * 0.5  # Half-spread in bps
        
        # Estimate based on order type
        if order.order_type == "market":
            return 2.0  # Assume 2 bps for market orders
        elif order.order_type == "limit":
            return 0.5  # Lower cost for limit orders
        return 1.0  # Default
        
    def _calculate_impact_cost(
        self, 
        order: ChildOrder, 
        fills: list[Fill], 
        market_data: dict[str, Any] | None
    ) -> float:
        """Calculate market impact cost in bps using square-root model"""
        total_qty = sum(fill.qty for fill in fills)
        
        # Get market volume if available
        if market_data and "volume" in market_data:
            adv = market_data["volume"]  # Average daily volume
            participation = total_qty / adv if adv > 0 else 0.01
        else:
            participation = total_qty / 1_000_000  # Default 1M shares ADV
        
        # Square-root impact model: impact ~ sqrt(participation)
        # Coefficient calibrated for typical equity markets
        impact_coefficient = 10.0  # bps for 1% participation
        impact_bps = impact_coefficient * (participation ** 0.5)
        
        # Cap at reasonable maximum
        return min(10.0, impact_bps)
        
    def _calculate_timing_cost(
        self, 
        order: ChildOrder, 
        fills: list[Fill], 
        market_data: dict[str, Any] | None
    ) -> float:
        """Calculate timing cost based on execution duration"""
        if not fills:
            return 0.0
            
        # Calculate execution duration
        first_fill_time = min(fill.ts for fill in fills)
        last_fill_time = max(fill.ts for fill in fills)
        duration_seconds = (last_fill_time - first_fill_time).total_seconds()
        
        # Get volatility if available
        if market_data and "volatility" in market_data:
            daily_vol = market_data["volatility"]
        else:
            daily_vol = 0.02  # Default 2% daily volatility
        
        # Convert to per-second volatility
        # Assuming 6.5 trading hours = 23400 seconds
        seconds_per_day = 23400
        vol_per_second = daily_vol / (seconds_per_day ** 0.5)
        
        # Timing cost = volatility * sqrt(duration)
        timing_cost_pct = vol_per_second * (duration_seconds ** 0.5)
        
        return timing_cost_pct * 10000  # Convert to bps
        
    def get_summary_stats(self) -> dict[str, float]:
        """Get summary TCA statistics"""
        if not self.reports:
            return {}
            
        return {
            "avg_total_cost_bps": sum(r.total_cost_bps for r in self.reports) / len(self.reports),
            "avg_spread_cost_bps": sum(r.spread_cost_bps for r in self.reports) / len(self.reports),
            "avg_impact_cost_bps": sum(r.impact_cost_bps for r in self.reports) / len(self.reports),
            "avg_timing_cost_bps": sum(r.timing_cost_bps for r in self.reports) / len(self.reports),
            "total_executions": len(self.reports)
        }
