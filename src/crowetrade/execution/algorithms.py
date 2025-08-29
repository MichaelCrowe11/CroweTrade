"""Execution algorithms for optimal order placement."""

from __future__ import annotations

import asyncio
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from crowetrade.core.types import OrderId
from crowetrade.execution.brokers.base import Order, OrderType

logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Execution plan for an algorithm."""
    
    symbol: str
    total_quantity: float
    side: str  # "buy" or "sell"
    start_time: datetime
    end_time: datetime
    slices: List[Tuple[datetime, float]]  # (time, quantity) pairs
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ExecutionAlgorithm(ABC):
    """Base class for execution algorithms."""
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        side: str,
        duration_minutes: int = 60,
    ):
        """Initialize algorithm.
        
        Args:
            symbol: Symbol to trade
            quantity: Total quantity to execute
            side: "buy" or "sell"
            duration_minutes: Time window for execution
        """
        self.symbol = symbol
        self.quantity = quantity
        self.side = side
        self.duration_minutes = duration_minutes
        self.executed_quantity = 0.0
        self.remaining_quantity = quantity
    
    @abstractmethod
    def generate_plan(self, market_data: Dict[str, float]) -> ExecutionPlan:
        """Generate execution plan based on market data."""
        pass
    
    @abstractmethod
    def adjust_plan(
        self,
        plan: ExecutionPlan,
        market_data: Dict[str, float],
        filled_quantity: float,
    ) -> ExecutionPlan:
        """Adjust plan based on execution progress."""
        pass


class TWAPAlgorithm(ExecutionAlgorithm):
    """Time-Weighted Average Price algorithm.
    
    Executes orders evenly over time to achieve average price
    over the execution window.
    """
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        side: str,
        duration_minutes: int = 60,
        min_slice_size: float = 100,
        randomize: bool = True,
        randomize_pct: float = 0.2,
    ):
        """Initialize TWAP algorithm.
        
        Args:
            symbol: Symbol to trade
            quantity: Total quantity to execute
            side: "buy" or "sell"
            duration_minutes: Time window for execution
            min_slice_size: Minimum order size per slice
            randomize: Whether to randomize slice sizes
            randomize_pct: Randomization percentage (0.2 = ±20%)
        """
        super().__init__(symbol, quantity, side, duration_minutes)
        self.min_slice_size = min_slice_size
        self.randomize = randomize
        self.randomize_pct = randomize_pct
    
    def generate_plan(self, market_data: Dict[str, float]) -> ExecutionPlan:
        """Generate TWAP execution plan."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=self.duration_minutes)
        
        # Calculate number of slices
        num_slices = max(
            1,
            min(
                self.duration_minutes,  # At most one per minute
                int(self.quantity / self.min_slice_size),
            ),
        )
        
        # Calculate base slice size
        base_slice_size = self.quantity / num_slices
        
        # Generate time points
        time_interval = self.duration_minutes / num_slices
        slices = []
        
        for i in range(num_slices):
            # Calculate time for this slice
            slice_time = start_time + timedelta(minutes=i * time_interval)
            
            # Calculate quantity for this slice
            if self.randomize and num_slices > 1:
                # Add randomization
                rand_factor = 1.0 + np.random.uniform(
                    -self.randomize_pct,
                    self.randomize_pct,
                )
                slice_quantity = base_slice_size * rand_factor
            else:
                slice_quantity = base_slice_size
            
            # Ensure minimum size
            slice_quantity = max(self.min_slice_size, slice_quantity)
            
            slices.append((slice_time, slice_quantity))
        
        # Adjust last slice to ensure total quantity matches
        if slices:
            total_planned = sum(q for _, q in slices[:-1])
            remaining = self.quantity - total_planned
            if remaining > 0:
                slices[-1] = (slices[-1][0], remaining)
        
        return ExecutionPlan(
            symbol=self.symbol,
            total_quantity=self.quantity,
            side=self.side,
            start_time=start_time,
            end_time=end_time,
            slices=slices,
            metadata={"algorithm": "TWAP", "num_slices": num_slices},
        )
    
    def adjust_plan(
        self,
        plan: ExecutionPlan,
        market_data: Dict[str, float],
        filled_quantity: float,
    ) -> ExecutionPlan:
        """Adjust TWAP plan based on progress."""
        # Calculate remaining quantity
        remaining = self.quantity - filled_quantity
        if remaining <= 0:
            return plan
        
        # Find remaining slices
        current_time = datetime.utcnow()
        remaining_slices = [
            (t, q) for t, q in plan.slices if t > current_time
        ]
        
        if not remaining_slices:
            # No time left, create one final slice
            plan.slices = [(current_time, remaining)]
        else:
            # Redistribute remaining quantity
            num_remaining = len(remaining_slices)
            new_slice_size = remaining / num_remaining
            
            plan.slices = [
                (t, q) for t, q in plan.slices if t <= current_time
            ] + [(t, new_slice_size) for t, _ in remaining_slices]
        
        return plan


class VWAPAlgorithm(ExecutionAlgorithm):
    """Volume-Weighted Average Price algorithm.
    
    Executes orders following historical volume patterns to minimize
    market impact and achieve VWAP.
    """
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        side: str,
        duration_minutes: int = 60,
        volume_profile: Optional[List[float]] = None,
        participation_rate: float = 0.1,
        min_slice_size: float = 100,
    ):
        """Initialize VWAP algorithm.
        
        Args:
            symbol: Symbol to trade
            quantity: Total quantity to execute
            side: "buy" or "sell"
            duration_minutes: Time window for execution
            volume_profile: Historical volume distribution (if None, use uniform)
            participation_rate: Maximum participation rate (0.1 = 10% of volume)
            min_slice_size: Minimum order size per slice
        """
        super().__init__(symbol, quantity, side, duration_minutes)
        self.volume_profile = volume_profile or self._default_volume_profile()
        self.participation_rate = participation_rate
        self.min_slice_size = min_slice_size
    
    def _default_volume_profile(self) -> List[float]:
        """Generate default U-shaped volume profile."""
        # U-shaped profile: higher at open/close
        num_buckets = min(self.duration_minutes, 30)
        x = np.linspace(0, 1, num_buckets)
        profile = 2.0 - 4.0 * (x - 0.5) ** 2  # U-shape
        return (profile / profile.sum()).tolist()
    
    def generate_plan(self, market_data: Dict[str, float]) -> ExecutionPlan:
        """Generate VWAP execution plan."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=self.duration_minutes)
        
        # Get expected volume from market data
        daily_volume = market_data.get("volume", 1_000_000)
        expected_period_volume = daily_volume * (self.duration_minutes / 390)  # Trading minutes
        
        # Calculate our participation
        max_participation = expected_period_volume * self.participation_rate
        
        # Adjust quantity if needed
        planned_quantity = min(self.quantity, max_participation)
        
        # Distribute quantity according to volume profile
        num_slices = len(self.volume_profile)
        time_interval = self.duration_minutes / num_slices
        
        slices = []
        for i, vol_weight in enumerate(self.volume_profile):
            slice_time = start_time + timedelta(minutes=i * time_interval)
            slice_quantity = planned_quantity * vol_weight
            
            # Ensure minimum size
            if slice_quantity < self.min_slice_size and slice_quantity > 0:
                slice_quantity = self.min_slice_size
            
            if slice_quantity > 0:
                slices.append((slice_time, slice_quantity))
        
        return ExecutionPlan(
            symbol=self.symbol,
            total_quantity=planned_quantity,
            side=self.side,
            start_time=start_time,
            end_time=end_time,
            slices=slices,
            metadata={
                "algorithm": "VWAP",
                "participation_rate": self.participation_rate,
                "expected_volume": expected_period_volume,
            },
        )
    
    def adjust_plan(
        self,
        plan: ExecutionPlan,
        market_data: Dict[str, float],
        filled_quantity: float,
    ) -> ExecutionPlan:
        """Adjust VWAP plan based on actual volume."""
        # Get actual volume traded
        current_volume = market_data.get("volume", 0)
        
        # Calculate remaining quantity
        remaining = self.quantity - filled_quantity
        if remaining <= 0:
            return plan
        
        # Find remaining slices
        current_time = datetime.utcnow()
        remaining_slices = [
            (t, q) for t, q in plan.slices if t > current_time
        ]
        
        if not remaining_slices:
            # No time left, create final slice
            plan.slices = [(current_time, remaining)]
        else:
            # Adjust based on actual vs expected volume
            expected_volume = plan.metadata.get("expected_volume", current_volume)
            volume_ratio = current_volume / expected_volume if expected_volume > 0 else 1.0
            
            # If volume is higher than expected, we can be more aggressive
            # If lower, we need to be more conservative
            adjustment_factor = min(2.0, max(0.5, volume_ratio))
            
            # Redistribute remaining quantity
            total_weight = sum(q for _, q in remaining_slices)
            if total_weight > 0:
                plan.slices = [
                    (t, q) for t, q in plan.slices if t <= current_time
                ] + [
                    (t, (q / total_weight) * remaining * adjustment_factor)
                    for t, q in remaining_slices
                ]
        
        return plan


class POVAlgorithm(ExecutionAlgorithm):
    """Percentage of Volume algorithm.
    
    Maintains a specified participation rate relative to market volume.
    """
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        side: str,
        duration_minutes: int = 60,
        target_pov: float = 0.1,
        min_pov: float = 0.05,
        max_pov: float = 0.25,
        min_slice_size: float = 100,
    ):
        """Initialize POV algorithm.
        
        Args:
            symbol: Symbol to trade
            quantity: Total quantity to execute
            side: "buy" or "sell"
            duration_minutes: Time window for execution
            target_pov: Target percentage of volume (0.1 = 10%)
            min_pov: Minimum POV
            max_pov: Maximum POV
            min_slice_size: Minimum order size
        """
        super().__init__(symbol, quantity, side, duration_minutes)
        self.target_pov = target_pov
        self.min_pov = min_pov
        self.max_pov = max_pov
        self.min_slice_size = min_slice_size
        self.executed_volume = 0.0
        self.market_volume = 0.0
    
    def generate_plan(self, market_data: Dict[str, float]) -> ExecutionPlan:
        """Generate POV execution plan."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=self.duration_minutes)
        
        # Estimate volume rate
        daily_volume = market_data.get("volume", 1_000_000)
        volume_per_minute = daily_volume / 390  # Trading minutes
        
        # Calculate slices based on expected volume
        slices = []
        remaining = self.quantity
        
        for i in range(self.duration_minutes):
            if remaining <= 0:
                break
            
            slice_time = start_time + timedelta(minutes=i)
            
            # Calculate slice size based on POV
            expected_volume = volume_per_minute
            slice_quantity = min(
                remaining,
                max(
                    self.min_slice_size,
                    expected_volume * self.target_pov,
                ),
            )
            
            slices.append((slice_time, slice_quantity))
            remaining -= slice_quantity
        
        return ExecutionPlan(
            symbol=self.symbol,
            total_quantity=self.quantity,
            side=self.side,
            start_time=start_time,
            end_time=end_time,
            slices=slices,
            metadata={
                "algorithm": "POV",
                "target_pov": self.target_pov,
                "volume_per_minute": volume_per_minute,
            },
        )
    
    def adjust_plan(
        self,
        plan: ExecutionPlan,
        market_data: Dict[str, float],
        filled_quantity: float,
    ) -> ExecutionPlan:
        """Adjust POV plan to maintain participation rate."""
        # Update market volume tracking
        current_volume = market_data.get("volume", 0)
        volume_delta = current_volume - self.market_volume
        self.market_volume = current_volume
        
        # Calculate current POV
        if volume_delta > 0:
            current_pov = filled_quantity / current_volume
        else:
            current_pov = self.target_pov
        
        # Adjust participation rate
        if current_pov < self.min_pov:
            # We're behind, be more aggressive
            adjusted_pov = min(self.max_pov, self.target_pov * 1.5)
        elif current_pov > self.max_pov:
            # We're ahead, slow down
            adjusted_pov = max(self.min_pov, self.target_pov * 0.5)
        else:
            adjusted_pov = self.target_pov
        
        # Calculate next slice
        remaining = self.quantity - filled_quantity
        if remaining <= 0:
            return plan
        
        current_time = datetime.utcnow()
        next_slice_quantity = min(
            remaining,
            max(self.min_slice_size, volume_delta * adjusted_pov),
        )
        
        # Update plan
        plan.slices = [
            (t, q) for t, q in plan.slices if t <= current_time
        ] + [(current_time + timedelta(seconds=30), next_slice_quantity)]
        
        return plan


class ImplementationShortfallAlgorithm(ExecutionAlgorithm):
    """Implementation Shortfall (IS) algorithm.
    
    Minimizes the difference between decision price and execution price
    by balancing market impact against timing risk.
    """
    
    def __init__(
        self,
        symbol: str,
        quantity: float,
        side: str,
        duration_minutes: int = 60,
        urgency: str = "normal",
        risk_aversion: float = 0.5,
        min_slice_size: float = 100,
    ):
        """Initialize IS algorithm.
        
        Args:
            symbol: Symbol to trade
            quantity: Total quantity to execute
            side: "buy" or "sell"
            duration_minutes: Time window for execution
            urgency: "low", "normal", "high", or "urgent"
            risk_aversion: Risk aversion parameter (0 = risk neutral, 1 = risk averse)
            min_slice_size: Minimum order size
        """
        super().__init__(symbol, quantity, side, duration_minutes)
        self.urgency = urgency
        self.risk_aversion = risk_aversion
        self.min_slice_size = min_slice_size
        self.decision_price = None
    
    def _urgency_to_rate(self) -> float:
        """Convert urgency to execution rate."""
        urgency_map = {
            "low": 0.2,
            "normal": 0.5,
            "high": 0.8,
            "urgent": 0.95,
        }
        return urgency_map.get(self.urgency, 0.5)
    
    def generate_plan(self, market_data: Dict[str, float]) -> ExecutionPlan:
        """Generate IS execution plan using Almgren-Chriss framework."""
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=self.duration_minutes)
        
        # Store decision price
        self.decision_price = market_data.get("last", 100.0)
        
        # Get volatility estimate
        volatility = market_data.get("volatility", 0.02)
        
        # Calculate optimal execution trajectory
        # Using simplified Almgren-Chriss solution
        urgency_rate = self._urgency_to_rate()
        kappa = urgency_rate * math.sqrt(self.risk_aversion / max(0.01, volatility))
        
        # Generate exponential decay schedule
        num_slices = min(self.duration_minutes, 20)
        slices = []
        remaining = self.quantity
        
        for i in range(num_slices):
            slice_time = start_time + timedelta(
                minutes=i * self.duration_minutes / num_slices
            )
            
            # Exponential decay
            decay_factor = math.exp(-kappa * i / num_slices)
            slice_quantity = remaining * (1 - math.exp(-kappa / num_slices))
            slice_quantity = max(self.min_slice_size, slice_quantity)
            
            if slice_quantity > 0 and remaining > 0:
                slice_quantity = min(slice_quantity, remaining)
                slices.append((slice_time, slice_quantity))
                remaining -= slice_quantity
        
        # Add final slice for any remaining
        if remaining > self.min_slice_size:
            slices.append((end_time, remaining))
        
        return ExecutionPlan(
            symbol=self.symbol,
            total_quantity=self.quantity,
            side=self.side,
            start_time=start_time,
            end_time=end_time,
            slices=slices,
            metadata={
                "algorithm": "IS",
                "urgency": self.urgency,
                "risk_aversion": self.risk_aversion,
                "decision_price": self.decision_price,
                "kappa": kappa,
            },
        )
    
    def adjust_plan(
        self,
        plan: ExecutionPlan,
        market_data: Dict[str, float],
        filled_quantity: float,
    ) -> ExecutionPlan:
        """Adjust IS plan based on market conditions."""
        current_price = market_data.get("last", self.decision_price)
        
        # Calculate implementation shortfall so far
        if self.decision_price:
            if self.side == "buy":
                shortfall = (current_price - self.decision_price) / self.decision_price
            else:
                shortfall = (self.decision_price - current_price) / self.decision_price
        else:
            shortfall = 0.0
        
        # Adjust urgency based on shortfall
        if shortfall > 0.01:  # Losing money
            # Increase urgency
            self.urgency = "high" if self.urgency == "normal" else "urgent"
        elif shortfall < -0.01:  # Making money
            # Decrease urgency
            self.urgency = "low" if self.urgency == "normal" else "normal"
        
        # Regenerate plan with new urgency
        new_plan = self.generate_plan(market_data)
        
        # Adjust for already filled quantity
        remaining = self.quantity - filled_quantity
        if remaining <= 0:
            return plan
        
        # Scale down slices
        scale_factor = remaining / self.quantity
        new_plan.slices = [
            (t, q * scale_factor) for t, q in new_plan.slices
        ]
        
        return new_plan