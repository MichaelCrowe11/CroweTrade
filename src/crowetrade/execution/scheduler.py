"""Execution Scheduler Module

Implements algorithmic execution strategies for optimal order placement:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price) 
- POV (Percentage of Volume)
- Implementation Shortfall (IS)
- Adaptive algorithms based on market microstructure

Each strategy aims to minimize market impact while achieving target
positions within specified time windows and risk constraints.
"""

import math
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Execution algorithm types"""
    TWAP = "twap"                    # Time-Weighted Average Price
    VWAP = "vwap"                    # Volume-Weighted Average Price
    POV = "pov"                      # Percentage of Volume
    IMPLEMENTATION_SHORTFALL = "is"  # Implementation Shortfall
    ADAPTIVE = "adaptive"            # Adaptive based on market conditions


@dataclass
class ExecutionParams:
    """Parameters for execution algorithms"""
    strategy: ExecutionStrategy
    target_quantity: float           # Total shares to execute
    time_horizon: timedelta         # Total execution time window
    max_participation_rate: float   # Max % of market volume (0.0-1.0)
    urgency: float                  # Urgency factor (0.0-1.0, higher = more aggressive)
    risk_aversion: float            # Risk aversion parameter for IS
    price_limit: Optional[float] = None      # Optional price limit
    min_fill_size: float = 1.0      # Minimum fill size
    max_slice_size: Optional[float] = None   # Maximum slice size


@dataclass
class ExecutionSlice:
    """Individual execution slice/order"""
    quantity: float                 # Shares to execute in this slice
    target_time: datetime          # Target execution time
    limit_price: Optional[float]   # Limit price (None for market orders)
    slice_id: str                  # Unique slice identifier
    parent_order_id: str           # Parent order reference
    urgency_factor: float          # Urgency for this specific slice


@dataclass
class ExecutionPlan:
    """Complete execution plan with all slices"""
    strategy: ExecutionStrategy
    total_quantity: float
    slices: List[ExecutionSlice]
    start_time: datetime
    end_time: datetime
    expected_cost: float           # Expected transaction cost
    expected_risk: float           # Expected execution risk (volatility)
    metadata: Dict[str, Any] = None


class ExecutionScheduler:
    """Advanced execution scheduling with multiple algorithms"""
    
    def __init__(self, 
                 default_participation_rate: float = 0.10,
                 market_impact_coefficient: float = 0.1,
                 temporary_impact_coefficient: float = 0.5,
                 volatility_scaling: float = 1.0):
        
        self.default_participation_rate = default_participation_rate
        self.market_impact_coefficient = market_impact_coefficient
        self.temporary_impact_coefficient = temporary_impact_coefficient
        self.volatility_scaling = volatility_scaling
        
        # Strategy implementations
        self.strategy_implementations = {
            ExecutionStrategy.TWAP: self._generate_twap_schedule,
            ExecutionStrategy.VWAP: self._generate_vwap_schedule,
            ExecutionStrategy.POV: self._generate_pov_schedule,
            ExecutionStrategy.IMPLEMENTATION_SHORTFALL: self._generate_is_schedule,
            ExecutionStrategy.ADAPTIVE: self._generate_adaptive_schedule
        }
        
        logger.info("ExecutionScheduler initialized")
    
    def create_execution_plan(self,
                            symbol: str,
                            params: ExecutionParams,
                            market_data: Dict[str, Any],
                            start_time: Optional[datetime] = None) -> ExecutionPlan:
        """Create optimal execution plan based on strategy and market conditions
        
        Args:
            symbol: Asset symbol to execute
            params: Execution parameters
            market_data: Current market data (price, volume, volatility, etc.)
            start_time: Execution start time (default: now)
            
        Returns:
            ExecutionPlan with optimized slices
        """
        if start_time is None:
            start_time = datetime.utcnow()
            
        end_time = start_time + params.time_horizon
        
        # Get strategy implementation
        strategy_fn = self.strategy_implementations.get(
            params.strategy, 
            self._generate_twap_schedule
        )
        
        try:
            # Generate execution slices
            slices = strategy_fn(symbol, params, market_data, start_time, end_time)
            
            # Calculate expected costs and risks
            expected_cost = self._calculate_expected_cost(slices, market_data)
            expected_risk = self._calculate_expected_risk(slices, market_data)
            
            plan = ExecutionPlan(
                strategy=params.strategy,
                total_quantity=params.target_quantity,
                slices=slices,
                start_time=start_time,
                end_time=end_time,
                expected_cost=expected_cost,
                expected_risk=expected_risk,
                metadata={
                    "symbol": symbol,
                    "num_slices": len(slices),
                    "avg_slice_size": params.target_quantity / len(slices) if slices else 0,
                    "participation_rate": params.max_participation_rate,
                    "created_at": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(f"Created {params.strategy.value} execution plan for {symbol}: "
                       f"{len(slices)} slices, expected cost: {expected_cost:.4f}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating execution plan for {symbol}: {e}")
            # Fallback to simple TWAP
            return self._create_fallback_plan(symbol, params, start_time, end_time)
    
    def _generate_twap_schedule(self, 
                              symbol: str,
                              params: ExecutionParams,
                              market_data: Dict[str, Any],
                              start_time: datetime,
                              end_time: datetime) -> List[ExecutionSlice]:
        """Generate Time-Weighted Average Price execution schedule"""
        
        # Calculate number of slices based on time horizon
        horizon_minutes = params.time_horizon.total_seconds() / 60
        num_slices = max(1, int(horizon_minutes / 5))  # 5-minute slices minimum
        
        # Apply max slice size constraint
        if params.max_slice_size:
            max_slices_by_size = math.ceil(abs(params.target_quantity) / params.max_slice_size)
            num_slices = min(num_slices, max_slices_by_size)
        
        slice_size = params.target_quantity / num_slices
        slice_duration = params.time_horizon / num_slices
        
        slices = []
        for i in range(num_slices):
            slice_time = start_time + (slice_duration * i)
            
            # Apply urgency factor to slice sizing
            urgency_adjustment = 1.0 + (params.urgency - 0.5) * 0.2 * (i / num_slices)
            adjusted_size = slice_size * urgency_adjustment
            
            slice_obj = ExecutionSlice(
                quantity=adjusted_size,
                target_time=slice_time,
                limit_price=params.price_limit,
                slice_id=f"{symbol}_twap_{i}_{int(slice_time.timestamp())}",
                parent_order_id=f"{symbol}_twap_{int(start_time.timestamp())}",
                urgency_factor=urgency_adjustment
            )
            slices.append(slice_obj)
        
        return slices
    
    def _generate_vwap_schedule(self,
                              symbol: str,
                              params: ExecutionParams,
                              market_data: Dict[str, Any],
                              start_time: datetime,
                              end_time: datetime) -> List[ExecutionSlice]:
        """Generate Volume-Weighted Average Price execution schedule"""
        
        # Get historical volume pattern (simplified)
        volume_profile = self._get_volume_profile(symbol, market_data)
        
        # Calculate slices weighted by expected volume
        total_expected_volume = sum(volume_profile)
        target_participation = min(params.max_participation_rate, 0.20)  # Cap at 20%
        
        slices = []
        slice_duration_minutes = 5  # 5-minute slices
        num_slices = int(params.time_horizon.total_seconds() / 60 / slice_duration_minutes)
        
        remaining_quantity = params.target_quantity
        
        for i in range(num_slices):
            slice_time = start_time + timedelta(minutes=i * slice_duration_minutes)
            
            # Volume weight for this time slice
            volume_weight = volume_profile[i % len(volume_profile)] if volume_profile else 1.0
            expected_volume = volume_weight * target_participation
            
            # Size slice based on volume and remaining quantity
            volume_based_size = expected_volume * target_participation
            slice_size = min(volume_based_size, abs(remaining_quantity) * 0.5)
            
            # Ensure we don't exceed target
            if abs(slice_size) > abs(remaining_quantity):
                slice_size = remaining_quantity
            
            if abs(slice_size) >= params.min_fill_size:
                slice_obj = ExecutionSlice(
                    quantity=slice_size,
                    target_time=slice_time,
                    limit_price=params.price_limit,
                    slice_id=f"{symbol}_vwap_{i}_{int(slice_time.timestamp())}",
                    parent_order_id=f"{symbol}_vwap_{int(start_time.timestamp())}",
                    urgency_factor=params.urgency
                )
                slices.append(slice_obj)
                remaining_quantity -= slice_size
            
            if abs(remaining_quantity) < params.min_fill_size:
                break
        
        # Handle any remaining quantity in final slice
        if abs(remaining_quantity) >= params.min_fill_size:
            final_slice = ExecutionSlice(
                quantity=remaining_quantity,
                target_time=end_time,
                limit_price=params.price_limit,
                slice_id=f"{symbol}_vwap_final_{int(end_time.timestamp())}",
                parent_order_id=f"{symbol}_vwap_{int(start_time.timestamp())}",
                urgency_factor=min(1.0, params.urgency * 1.5)
            )
            slices.append(final_slice)
        
        return slices
    
    def _generate_pov_schedule(self,
                             symbol: str,
                             params: ExecutionParams,
                             market_data: Dict[str, Any],
                             start_time: datetime,
                             end_time: datetime) -> List[ExecutionSlice]:
        """Generate Percentage of Volume execution schedule"""
        
        # Estimate market volume rate
        current_volume = market_data.get("volume", 1000000)
        time_elapsed = market_data.get("time_elapsed_today", timedelta(hours=1))
        volume_rate = current_volume / time_elapsed.total_seconds() * 60  # per minute
        
        target_rate = volume_rate * params.max_participation_rate
        slice_duration_minutes = 2  # More frequent slices for POV
        
        num_slices = int(params.time_horizon.total_seconds() / 60 / slice_duration_minutes)
        slice_quantity = target_rate * slice_duration_minutes
        
        slices = []
        remaining_quantity = params.target_quantity
        
        for i in range(num_slices):
            slice_time = start_time + timedelta(minutes=i * slice_duration_minutes)
            
            # Adjust slice size based on remaining quantity
            current_slice_size = min(slice_quantity, abs(remaining_quantity))
            
            # Apply urgency scaling
            urgency_factor = 1.0 + params.urgency * 0.5
            current_slice_size *= urgency_factor
            
            # Maintain sign of target quantity
            if params.target_quantity < 0:
                current_slice_size = -current_slice_size
            
            if abs(current_slice_size) >= params.min_fill_size:
                slice_obj = ExecutionSlice(
                    quantity=current_slice_size,
                    target_time=slice_time,
                    limit_price=params.price_limit,
                    slice_id=f"{symbol}_pov_{i}_{int(slice_time.timestamp())}",
                    parent_order_id=f"{symbol}_pov_{int(start_time.timestamp())}",
                    urgency_factor=urgency_factor
                )
                slices.append(slice_obj)
                remaining_quantity -= current_slice_size
            
            if abs(remaining_quantity) < params.min_fill_size:
                break
        
        return slices
    
    def _generate_is_schedule(self,
                            symbol: str,
                            params: ExecutionParams,
                            market_data: Dict[str, Any],
                            start_time: datetime,
                            end_time: datetime) -> List[ExecutionSlice]:
        """Generate Implementation Shortfall optimized schedule"""
        
        # Implementation Shortfall parameters
        volatility = market_data.get("volatility", 0.02)  # Daily volatility
        mid_price = market_data.get("mid_price", market_data.get("price", 100.0))
        
        # Almgren-Chriss style optimization (simplified)
        T = params.time_horizon.total_seconds() / (252 * 24 * 3600)  # Time in years
        Q = abs(params.target_quantity)
        
        # Market impact parameters
        eta = self.temporary_impact_coefficient  # Temporary impact
        gamma = self.market_impact_coefficient   # Permanent impact
        
        # Optimal execution rate (Almgren-Chriss)
        kappa = math.sqrt(params.risk_aversion * volatility**2 / eta)
        
        # Number of slices
        num_slices = max(5, int(math.sqrt(Q) / 10))  # Adaptive based on order size
        dt = T / num_slices
        
        slices = []
        remaining_quantity = params.target_quantity
        
        for i in range(num_slices):
            slice_time = start_time + timedelta(seconds=(T * (252 * 24 * 3600) * i / num_slices))
            
            # Optimal trajectory (exponential decay)
            t_remaining = (num_slices - i) / num_slices
            optimal_size = remaining_quantity * (1 - math.exp(-kappa * dt)) / (1 - math.exp(-kappa * t_remaining))
            
            # Apply constraints
            optimal_size = max(params.min_fill_size, min(abs(optimal_size), abs(remaining_quantity)))
            
            if params.target_quantity < 0:
                optimal_size = -optimal_size
            
            # Calculate limit price based on urgency and market impact
            impact_adjustment = eta * abs(optimal_size) / Q
            if params.target_quantity > 0:  # Buying
                limit_price = mid_price * (1 + impact_adjustment * (2 - params.urgency))
            else:  # Selling
                limit_price = mid_price * (1 - impact_adjustment * (2 - params.urgency))
            
            slice_obj = ExecutionSlice(
                quantity=optimal_size,
                target_time=slice_time,
                limit_price=limit_price if not params.price_limit else params.price_limit,
                slice_id=f"{symbol}_is_{i}_{int(slice_time.timestamp())}",
                parent_order_id=f"{symbol}_is_{int(start_time.timestamp())}",
                urgency_factor=params.urgency
            )
            slices.append(slice_obj)
            remaining_quantity -= optimal_size
        
        return slices
    
    def _generate_adaptive_schedule(self,
                                  symbol: str,
                                  params: ExecutionParams,
                                  market_data: Dict[str, Any],
                                  start_time: datetime,
                                  end_time: datetime) -> List[ExecutionSlice]:
        """Generate adaptive execution schedule based on market conditions"""
        
        # Analyze market conditions
        volatility = market_data.get("volatility", 0.02)
        volume = market_data.get("volume", 1000000)
        spread = market_data.get("spread", 0.01)
        
        # Choose base strategy based on market conditions
        if volatility > 0.03 and spread > 0.02:
            # High volatility, wide spreads -> Use IS algorithm
            base_params = params._replace(strategy=ExecutionStrategy.IMPLEMENTATION_SHORTFALL)
            return self._generate_is_schedule(symbol, base_params, market_data, start_time, end_time)
        elif volume > 5000000:
            # High volume -> Use VWAP
            base_params = params._replace(strategy=ExecutionStrategy.VWAP)
            return self._generate_vwap_schedule(symbol, base_params, market_data, start_time, end_time)
        else:
            # Normal conditions -> Use TWAP with POV constraints
            base_params = params._replace(strategy=ExecutionStrategy.TWAP)
            return self._generate_twap_schedule(symbol, base_params, market_data, start_time, end_time)
    
    def _get_volume_profile(self, symbol: str, market_data: Dict[str, Any]) -> List[float]:
        """Get intraday volume profile (simplified)"""
        # In production, this would use historical volume patterns
        # Simplified U-shaped intraday pattern
        return [1.5, 1.2, 0.8, 0.6, 0.5, 0.7, 0.9, 1.1, 1.3, 1.4]  # 10 time buckets
    
    def _calculate_expected_cost(self, slices: List[ExecutionSlice], market_data: Dict[str, Any]) -> float:
        """Calculate expected transaction cost for execution plan"""
        if not slices:
            return 0.0
            
        total_cost = 0.0
        mid_price = market_data.get("mid_price", market_data.get("price", 100.0))
        spread = market_data.get("spread", 0.01)
        
        for slice_obj in slices:
            # Market impact cost (temporary + permanent)
            size_impact = self.temporary_impact_coefficient * math.sqrt(abs(slice_obj.quantity))
            spread_cost = spread / 2  # Half-spread cost
            
            slice_cost = (size_impact + spread_cost) / mid_price
            total_cost += abs(slice_obj.quantity) * slice_cost
        
        return total_cost
    
    def _calculate_expected_risk(self, slices: List[ExecutionSlice], market_data: Dict[str, Any]) -> float:
        """Calculate expected execution risk (volatility)"""
        if not slices:
            return 0.0
            
        volatility = market_data.get("volatility", 0.02)
        total_quantity = sum(abs(s.quantity) for s in slices)
        
        # Risk increases with execution time and quantity
        execution_time_hours = len(slices) * 0.1  # Approximate
        time_risk = volatility * math.sqrt(execution_time_hours / 24)  # Scale to execution period
        
        return time_risk
    
    def _create_fallback_plan(self,
                            symbol: str,
                            params: ExecutionParams,
                            start_time: datetime,
                            end_time: datetime) -> ExecutionPlan:
        """Create simple fallback execution plan"""
        # Simple single slice execution
        slice_obj = ExecutionSlice(
            quantity=params.target_quantity,
            target_time=start_time,
            limit_price=params.price_limit,
            slice_id=f"{symbol}_fallback_{int(start_time.timestamp())}",
            parent_order_id=f"{symbol}_fallback_{int(start_time.timestamp())}",
            urgency_factor=1.0
        )
        
        return ExecutionPlan(
            strategy=ExecutionStrategy.TWAP,
            total_quantity=params.target_quantity,
            slices=[slice_obj],
            start_time=start_time,
            end_time=end_time,
            expected_cost=0.01,  # 1% placeholder cost
            expected_risk=0.02,   # 2% placeholder risk
            metadata={"fallback": True}
        )


def create_execution_params(strategy_name: str,
                          target_quantity: float,
                          time_horizon_minutes: int,
                          **kwargs) -> ExecutionParams:
    """Convenience function to create ExecutionParams"""
    
    strategy_map = {
        "twap": ExecutionStrategy.TWAP,
        "vwap": ExecutionStrategy.VWAP,
        "pov": ExecutionStrategy.POV,
        "is": ExecutionStrategy.IMPLEMENTATION_SHORTFALL,
        "adaptive": ExecutionStrategy.ADAPTIVE
    }
    
    strategy = strategy_map.get(strategy_name.lower(), ExecutionStrategy.TWAP)
    
    return ExecutionParams(
        strategy=strategy,
        target_quantity=target_quantity,
        time_horizon=timedelta(minutes=time_horizon_minutes),
        max_participation_rate=kwargs.get("max_participation_rate", 0.10),
        urgency=kwargs.get("urgency", 0.5),
        risk_aversion=kwargs.get("risk_aversion", 1.0),
        price_limit=kwargs.get("price_limit"),
        min_fill_size=kwargs.get("min_fill_size", 1.0),
        max_slice_size=kwargs.get("max_slice_size")
    )
