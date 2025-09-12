"""Advanced execution scheduler with market impact models and algo implementations"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """Execution algorithm types"""
    TWAP = "twap"           # Time-Weighted Average Price
    VWAP = "vwap"           # Volume-Weighted Average Price
    POV = "pov"             # Percentage of Volume
    IS = "is"               # Implementation Shortfall
    ICEBERG = "iceberg"     # Hidden size orders
    ADAPTIVE = "adaptive"   # ML-based adaptive execution


@dataclass
class MarketConditions:
    """Current market conditions for execution"""
    bid: float
    ask: float
    mid: float
    spread: float
    volume: float
    volatility: float
    average_daily_volume: float
    participation_rate: float
    liquidity_score: float
    timestamp: datetime


@dataclass
class OrderSlice:
    """A slice of a parent order for execution"""
    slice_id: str
    parent_order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    target_price: Optional[float]
    start_time: datetime
    end_time: datetime
    urgency: float  # 0-1 scale
    venue: Optional[str] = None
    executed_quantity: float = 0
    average_price: float = 0
    status: str = "pending"
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.executed_quantity
    
    @property
    def is_complete(self) -> bool:
        return self.remaining_quantity <= 0


@dataclass
class ExecutionPlan:
    """Complete execution plan for an order"""
    plan_id: str
    order_id: str
    symbol: str
    total_quantity: float
    algorithm: ExecutionAlgorithm
    slices: List[OrderSlice]
    estimated_cost: float
    estimated_impact: float
    risk_score: float
    created_at: datetime
    metadata: Dict[str, Any]


class MarketImpactModel:
    """Estimate market impact of trades"""
    
    def __init__(
        self,
        permanent_impact_factor: float = 0.1,
        temporary_impact_factor: float = 0.05,
        decay_rate: float = 0.5
    ):
        self.permanent_impact_factor = permanent_impact_factor
        self.temporary_impact_factor = temporary_impact_factor
        self.decay_rate = decay_rate
    
    def estimate_impact(
        self,
        quantity: float,
        adv: float,  # Average Daily Volume
        volatility: float,
        urgency: float = 0.5,
        spread: float = 0.001
    ) -> Dict[str, float]:
        """
        Estimate market impact using Almgren-Chriss model
        
        Returns:
            Dict with permanent and temporary impact estimates
        """
        
        # Participation rate
        participation = quantity / adv
        
        # Permanent impact (information leakage)
        permanent_impact = (
            self.permanent_impact_factor *
            volatility *
            np.sqrt(participation) *
            np.sign(quantity)
        )
        
        # Temporary impact (liquidity consumption)
        temporary_impact = (
            self.temporary_impact_factor *
            volatility *
            participation *
            urgency +
            0.5 * spread
        )
        
        # Total expected cost
        total_impact = abs(permanent_impact) + abs(temporary_impact)
        
        return {
            'permanent_impact': permanent_impact,
            'temporary_impact': temporary_impact,
            'total_impact': total_impact,
            'impact_percentage': total_impact * 100  # As percentage
        }


class ExecutionScheduler:
    """
    Advanced execution scheduler with multiple algorithms
    """
    
    def __init__(
        self,
        impact_model: Optional[MarketImpactModel] = None,
        default_participation_rate: float = 0.1,
        max_participation_rate: float = 0.3,
        min_slice_size: float = 100,
        adaptive_learning: bool = False
    ):
        self.impact_model = impact_model or MarketImpactModel()
        self.default_participation_rate = default_participation_rate
        self.max_participation_rate = max_participation_rate
        self.min_slice_size = min_slice_size
        self.adaptive_learning = adaptive_learning
        
        # Algorithm implementations
        self.algorithms = {
            ExecutionAlgorithm.TWAP: self._schedule_twap,
            ExecutionAlgorithm.VWAP: self._schedule_vwap,
            ExecutionAlgorithm.POV: self._schedule_pov,
            ExecutionAlgorithm.IS: self._schedule_is,
            ExecutionAlgorithm.ICEBERG: self._schedule_iceberg,
            ExecutionAlgorithm.ADAPTIVE: self._schedule_adaptive
        }
        
        # Execution history for learning
        self.execution_history: List[Dict[str, Any]] = []
    
    def create_execution_plan(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        algorithm: ExecutionAlgorithm,
        market_conditions: MarketConditions,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        urgency: float = 0.5,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        Create an execution plan for an order
        
        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Total quantity to execute
            algorithm: Execution algorithm to use
            market_conditions: Current market conditions
            start_time: Execution start time
            end_time: Execution end time
            urgency: Urgency level (0-1)
            constraints: Additional constraints
        
        Returns:
            ExecutionPlan with scheduled slices
        """
        
        # Default times if not provided
        if start_time is None:
            start_time = datetime.now()
        if end_time is None:
            end_time = start_time + timedelta(hours=1)
        
        # Get algorithm implementation
        scheduler_func = self.algorithms.get(algorithm)
        if not scheduler_func:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Create slices using selected algorithm
        slices = scheduler_func(
            symbol=symbol,
            side=side,
            quantity=quantity,
            market_conditions=market_conditions,
            start_time=start_time,
            end_time=end_time,
            urgency=urgency,
            constraints=constraints or {}
        )
        
        # Estimate execution costs
        impact_estimate = self.impact_model.estimate_impact(
            quantity=quantity,
            adv=market_conditions.average_daily_volume,
            volatility=market_conditions.volatility,
            urgency=urgency,
            spread=market_conditions.spread
        )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(
            quantity, market_conditions, urgency
        )
        
        # Create execution plan
        plan = ExecutionPlan(
            plan_id=f"plan_{order_id}",
            order_id=order_id,
            symbol=symbol,
            total_quantity=quantity,
            algorithm=algorithm,
            slices=slices,
            estimated_cost=impact_estimate['total_impact'] * quantity * market_conditions.mid,
            estimated_impact=impact_estimate['impact_percentage'],
            risk_score=risk_score,
            created_at=datetime.now(),
            metadata={
                'urgency': urgency,
                'market_conditions': {
                    'spread': market_conditions.spread,
                    'volatility': market_conditions.volatility,
                    'liquidity_score': market_conditions.liquidity_score
                }
            }
        )
        
        return plan
    
    def _schedule_twap(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_conditions: MarketConditions,
        start_time: datetime,
        end_time: datetime,
        urgency: float,
        constraints: Dict[str, Any]
    ) -> List[OrderSlice]:
        """Time-Weighted Average Price scheduling"""
        
        # Calculate time intervals
        duration = (end_time - start_time).total_seconds() / 60  # minutes
        num_slices = max(1, int(duration / 5))  # 5-minute intervals
        num_slices = min(num_slices, int(quantity / self.min_slice_size))
        
        if num_slices == 0:
            num_slices = 1
        
        # Equal-sized slices
        slice_quantity = quantity / num_slices
        time_interval = duration / num_slices
        
        slices = []
        for i in range(num_slices):
            slice_start = start_time + timedelta(minutes=i * time_interval)
            slice_end = start_time + timedelta(minutes=(i + 1) * time_interval)
            
            slice_obj = OrderSlice(
                slice_id=f"twap_{symbol}_{i}",
                parent_order_id=f"order_{symbol}",
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                target_price=None,  # Market orders
                start_time=slice_start,
                end_time=slice_end,
                urgency=urgency
            )
            
            slices.append(slice_obj)
        
        return slices
    
    def _schedule_vwap(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_conditions: MarketConditions,
        start_time: datetime,
        end_time: datetime,
        urgency: float,
        constraints: Dict[str, Any]
    ) -> List[OrderSlice]:
        """Volume-Weighted Average Price scheduling"""
        
        # Get historical volume profile (simplified U-shape)
        duration_hours = (end_time - start_time).total_seconds() / 3600
        num_slices = max(1, int(duration_hours * 4))  # 15-minute intervals
        
        # U-shaped volume profile (higher at open/close)
        volume_profile = self._get_volume_profile(num_slices)
        
        slices = []
        time_interval = (end_time - start_time) / num_slices
        
        for i in range(num_slices):
            # Allocate quantity based on volume profile
            slice_quantity = quantity * volume_profile[i]
            
            slice_start = start_time + (time_interval * i)
            slice_end = start_time + (time_interval * (i + 1))
            
            slice_obj = OrderSlice(
                slice_id=f"vwap_{symbol}_{i}",
                parent_order_id=f"order_{symbol}",
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                target_price=None,
                start_time=slice_start,
                end_time=slice_end,
                urgency=urgency * volume_profile[i]  # Adjust urgency by volume
            )
            
            slices.append(slice_obj)
        
        return slices
    
    def _schedule_pov(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_conditions: MarketConditions,
        start_time: datetime,
        end_time: datetime,
        urgency: float,
        constraints: Dict[str, Any]
    ) -> List[OrderSlice]:
        """Percentage of Volume scheduling"""
        
        # Target participation rate
        target_pov = constraints.get('target_pov', self.default_participation_rate)
        target_pov = min(target_pov, self.max_participation_rate)
        
        # Estimate total market volume during period
        duration_hours = (end_time - start_time).total_seconds() / 3600
        expected_volume = market_conditions.average_daily_volume * (duration_hours / 6.5)  # 6.5 trading hours
        
        # Calculate number of slices
        our_volume = quantity
        market_volume_per_slice = expected_volume * target_pov
        num_slices = max(1, int(our_volume / (market_volume_per_slice + 1e-9)))
        
        slices = []
        time_interval = (end_time - start_time) / num_slices
        
        for i in range(num_slices):
            slice_quantity = quantity / num_slices
            
            slice_start = start_time + (time_interval * i)
            slice_end = start_time + (time_interval * (i + 1))
            
            slice_obj = OrderSlice(
                slice_id=f"pov_{symbol}_{i}",
                parent_order_id=f"order_{symbol}",
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                target_price=None,
                start_time=slice_start,
                end_time=slice_end,
                urgency=urgency
            )
            
            slices.append(slice_obj)
        
        return slices
    
    def _schedule_is(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_conditions: MarketConditions,
        start_time: datetime,
        end_time: datetime,
        urgency: float,
        constraints: Dict[str, Any]
    ) -> List[OrderSlice]:
        """Implementation Shortfall minimization"""
        
        # Use Almgren-Chriss optimal execution
        duration = (end_time - start_time).total_seconds() / 60
        
        # Risk aversion parameter
        risk_aversion = constraints.get('risk_aversion', 1.0)
        
        # Calculate optimal trajectory
        n_slices = max(1, min(20, int(duration / 5)))
        times = np.linspace(0, duration, n_slices + 1)
        
        # Optimal execution rate (simplified)
        kappa = np.sqrt(risk_aversion * market_conditions.volatility)
        execution_rate = kappa * np.sinh(kappa * (duration - times[:-1])) / np.sinh(kappa * duration)
        
        # Normalize to sum to total quantity
        execution_rate = execution_rate / execution_rate.sum() * quantity
        
        slices = []
        for i in range(n_slices):
            slice_start = start_time + timedelta(minutes=times[i])
            slice_end = start_time + timedelta(minutes=times[i + 1])
            
            slice_obj = OrderSlice(
                slice_id=f"is_{symbol}_{i}",
                parent_order_id=f"order_{symbol}",
                symbol=symbol,
                side=side,
                quantity=execution_rate[i],
                target_price=market_conditions.mid,  # Track against arrival price
                start_time=slice_start,
                end_time=slice_end,
                urgency=urgency * (1 + i * 0.1)  # Increase urgency over time
            )
            
            slices.append(slice_obj)
        
        return slices
    
    def _schedule_iceberg(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_conditions: MarketConditions,
        start_time: datetime,
        end_time: datetime,
        urgency: float,
        constraints: Dict[str, Any]
    ) -> List[OrderSlice]:
        """Iceberg order scheduling (hidden size)"""
        
        # Visible size (typically 10-20% of average order size)
        visible_percentage = constraints.get('visible_percentage', 0.15)
        visible_size = market_conditions.average_daily_volume * 0.001 * visible_percentage
        
        # Create slices with visible size
        num_slices = max(1, int(quantity / visible_size))
        slice_quantity = quantity / num_slices
        
        # Random time intervals to avoid detection
        duration = (end_time - start_time).total_seconds() / 60
        
        slices = []
        current_time = start_time
        
        for i in range(num_slices):
            # Random interval (with minimum spacing)
            min_interval = duration / num_slices * 0.5
            max_interval = duration / num_slices * 1.5
            interval = np.random.uniform(min_interval, max_interval)
            
            slice_end = min(current_time + timedelta(minutes=interval), end_time)
            
            slice_obj = OrderSlice(
                slice_id=f"iceberg_{symbol}_{i}",
                parent_order_id=f"order_{symbol}",
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                target_price=None,
                start_time=current_time,
                end_time=slice_end,
                urgency=urgency
            )
            
            slices.append(slice_obj)
            current_time = slice_end
        
        return slices
    
    def _schedule_adaptive(
        self,
        symbol: str,
        side: str,
        quantity: float,
        market_conditions: MarketConditions,
        start_time: datetime,
        end_time: datetime,
        urgency: float,
        constraints: Dict[str, Any]
    ) -> List[OrderSlice]:
        """Adaptive ML-based scheduling"""
        
        # Start with TWAP and adapt based on conditions
        base_slices = self._schedule_twap(
            symbol, side, quantity, market_conditions,
            start_time, end_time, urgency, constraints
        )
        
        # Adjust based on market conditions
        if market_conditions.liquidity_score < 0.3:
            # Low liquidity: smaller slices, longer duration
            return self._adjust_for_low_liquidity(base_slices)
        
        elif market_conditions.volatility > 0.25:
            # High volatility: front-load execution
            return self._adjust_for_high_volatility(base_slices)
        
        elif market_conditions.spread > 0.002:
            # Wide spread: use limit orders
            return self._adjust_for_wide_spread(base_slices, market_conditions)
        
        return base_slices
    
    def _get_volume_profile(self, num_slices: int) -> np.ndarray:
        """Generate U-shaped intraday volume profile"""
        
        x = np.linspace(0, 1, num_slices)
        # U-shape: higher at boundaries
        profile = 1.5 - 2 * np.abs(x - 0.5)
        # Add noise
        profile += np.random.normal(0, 0.05, num_slices)
        # Normalize
        profile = np.maximum(profile, 0.1)
        profile = profile / profile.sum()
        
        return profile
    
    def _calculate_risk_score(
        self,
        quantity: float,
        market_conditions: MarketConditions,
        urgency: float
    ) -> float:
        """Calculate execution risk score (0-1)"""
        
        # Size risk
        size_risk = min(1.0, quantity / (market_conditions.average_daily_volume * 0.05))
        
        # Liquidity risk
        liquidity_risk = 1.0 - market_conditions.liquidity_score
        
        # Volatility risk
        vol_risk = min(1.0, market_conditions.volatility / 0.3)
        
        # Urgency risk
        urgency_risk = urgency
        
        # Weighted average
        risk_score = (
            size_risk * 0.3 +
            liquidity_risk * 0.3 +
            vol_risk * 0.2 +
            urgency_risk * 0.2
        )
        
        return min(1.0, risk_score)
    
    def _adjust_for_low_liquidity(
        self,
        slices: List[OrderSlice]
    ) -> List[OrderSlice]:
        """Adjust slices for low liquidity conditions"""
        
        # Reduce slice sizes and extend duration
        for slice_obj in slices:
            slice_obj.quantity *= 0.7
            # Extend end time
            duration = slice_obj.end_time - slice_obj.start_time
            slice_obj.end_time += duration * 0.5
        
        return slices
    
    def _adjust_for_high_volatility(
        self,
        slices: List[OrderSlice]
    ) -> List[OrderSlice]:
        """Adjust slices for high volatility"""
        
        # Front-load execution
        total_qty = sum(s.quantity for s in slices)
        
        for i, slice_obj in enumerate(slices):
            if i < len(slices) // 2:
                # First half gets more quantity
                slice_obj.quantity = total_qty * 0.7 / (len(slices) // 2)
            else:
                # Second half gets less
                slice_obj.quantity = total_qty * 0.3 / (len(slices) - len(slices) // 2)
        
        return slices
    
    def _adjust_for_wide_spread(
        self,
        slices: List[OrderSlice],
        market_conditions: MarketConditions
    ) -> List[OrderSlice]:
        """Adjust slices for wide spread"""
        
        # Use limit orders at mid or better
        for slice_obj in slices:
            if slice_obj.side == 'buy':
                slice_obj.target_price = market_conditions.mid - market_conditions.spread * 0.25
            else:
                slice_obj.target_price = market_conditions.mid + market_conditions.spread * 0.25
        
        return slices
    
    def update_execution_history(
        self,
        plan: ExecutionPlan,
        actual_impact: float,
        actual_cost: float
    ) -> None:
        """Update execution history for learning"""
        
        self.execution_history.append({
            'plan_id': plan.plan_id,
            'algorithm': plan.algorithm.value,
            'quantity': plan.total_quantity,
            'estimated_impact': plan.estimated_impact,
            'actual_impact': actual_impact,
            'estimated_cost': plan.estimated_cost,
            'actual_cost': actual_cost,
            'risk_score': plan.risk_score,
            'timestamp': datetime.now()
        })
        
        # Keep history bounded
        if len(self.execution_history) > 10000:
            self.execution_history.pop(0)
    
    def get_algorithm_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics by algorithm"""
        
        if not self.execution_history:
            return {}
        
        df = pd.DataFrame(self.execution_history)
        
        performance = {}
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            
            performance[algo] = {
                'count': len(algo_df),
                'avg_impact_error': (algo_df['actual_impact'] - algo_df['estimated_impact']).mean(),
                'avg_cost_error': (algo_df['actual_cost'] - algo_df['estimated_cost']).mean(),
                'impact_mae': abs(algo_df['actual_impact'] - algo_df['estimated_impact']).mean(),
                'cost_mae': abs(algo_df['actual_cost'] - algo_df['estimated_cost']).mean()
            }
        
        return performance