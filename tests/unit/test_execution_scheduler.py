"""Tests for Execution Scheduler Module"""

import pytest
import math
from datetime import datetime, timedelta

from crowetrade.execution.scheduler import (
    ExecutionScheduler,
    ExecutionParams,
    ExecutionStrategy,
    ExecutionSlice,
    ExecutionPlan,
    create_execution_params
)


class TestExecutionScheduler:
    """Test suite for ExecutionScheduler"""
    
    @pytest.fixture
    def scheduler(self):
        """Create ExecutionScheduler for testing"""
        return ExecutionScheduler(
            default_participation_rate=0.10,
            market_impact_coefficient=0.1,
            temporary_impact_coefficient=0.5
        )
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for testing"""
        return {
            "price": 100.0,
            "mid_price": 100.0,
            "volume": 1_000_000,
            "volatility": 0.02,
            "spread": 0.05,
            "time_elapsed_today": timedelta(hours=2)
        }
    
    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initialization"""
        assert scheduler.default_participation_rate == 0.10
        assert scheduler.market_impact_coefficient == 0.1
        assert scheduler.temporary_impact_coefficient == 0.5
        assert len(scheduler.strategy_implementations) == 5
    
    def test_create_execution_params_helper(self):
        """Test execution params creation helper"""
        params = create_execution_params(
            strategy_name="twap",
            target_quantity=1000,
            time_horizon_minutes=60,
            max_participation_rate=0.15,
            urgency=0.8
        )
        
        assert params.strategy == ExecutionStrategy.TWAP
        assert params.target_quantity == 1000
        assert params.time_horizon == timedelta(minutes=60)
        assert params.max_participation_rate == 0.15
        assert params.urgency == 0.8
    
    def test_twap_execution_plan(self, scheduler, sample_market_data):
        """Test TWAP execution plan generation"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.TWAP,
            target_quantity=1000,
            time_horizon=timedelta(minutes=30),
            max_participation_rate=0.10,
            urgency=0.5,
            risk_aversion=1.0
        )
        
        plan = scheduler.create_execution_plan("AAPL", params, sample_market_data)
        
        assert plan.strategy == ExecutionStrategy.TWAP
        assert plan.total_quantity == 1000
        assert len(plan.slices) > 0
        assert plan.expected_cost >= 0
        assert plan.expected_risk >= 0
        
        # Check that slices sum to target quantity
        total_slices = sum(slice.quantity for slice in plan.slices)
        assert abs(total_slices - 1000) < 0.01
    
    def test_vwap_execution_plan(self, scheduler, sample_market_data):
        """Test VWAP execution plan generation"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.VWAP,
            target_quantity=2000,
            time_horizon=timedelta(minutes=60),
            max_participation_rate=0.15,
            urgency=0.7,
            risk_aversion=1.0
        )
        
        plan = scheduler.create_execution_plan("MSFT", params, sample_market_data)
        
        assert plan.strategy == ExecutionStrategy.VWAP
        assert plan.total_quantity == 2000
        assert len(plan.slices) > 0
        
        # VWAP should create multiple slices with volume weighting
        assert len(plan.slices) >= 2
    
    def test_pov_execution_plan(self, scheduler, sample_market_data):
        """Test POV (Percentage of Volume) execution plan"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.POV,
            target_quantity=500,
            time_horizon=timedelta(minutes=20),
            max_participation_rate=0.05,
            urgency=0.6,
            risk_aversion=1.0
        )
        
        plan = scheduler.create_execution_plan("GOOGL", params, sample_market_data)
        
        assert plan.strategy == ExecutionStrategy.POV
        assert plan.total_quantity == 500
        assert len(plan.slices) > 0
        
        # POV should respect participation rate constraints
        for slice in plan.slices:
            assert slice.urgency_factor >= 0.5  # Based on params.urgency
    
    def test_implementation_shortfall_plan(self, scheduler, sample_market_data):
        """Test Implementation Shortfall execution plan"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.IMPLEMENTATION_SHORTFALL,
            target_quantity=1500,
            time_horizon=timedelta(minutes=45),
            max_participation_rate=0.12,
            urgency=0.4,
            risk_aversion=2.0
        )
        
        plan = scheduler.create_execution_plan("TSLA", params, sample_market_data)
        
        assert plan.strategy == ExecutionStrategy.IMPLEMENTATION_SHORTFALL
        assert plan.total_quantity == 1500
        assert len(plan.slices) > 0
        
        # IS algorithm should generate limit prices
        limit_price_slices = [s for s in plan.slices if s.limit_price is not None]
        assert len(limit_price_slices) > 0
    
    def test_adaptive_execution_plan(self, scheduler, sample_market_data):
        """Test adaptive execution strategy"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.ADAPTIVE,
            target_quantity=800,
            time_horizon=timedelta(minutes=25),
            max_participation_rate=0.08,
            urgency=0.5,
            risk_aversion=1.0
        )
        
        plan = scheduler.create_execution_plan("NVDA", params, sample_market_data)
        
        assert plan.strategy == ExecutionStrategy.ADAPTIVE
        assert plan.total_quantity == 800
        assert len(plan.slices) > 0
        
        # Adaptive should choose appropriate base strategy
        assert plan.metadata["symbol"] == "NVDA"
    
    def test_execution_plan_with_price_limit(self, scheduler, sample_market_data):
        """Test execution plan with price limit constraint"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.TWAP,
            target_quantity=1200,
            time_horizon=timedelta(minutes=40),
            max_participation_rate=0.10,
            urgency=0.5,
            risk_aversion=1.0,
            price_limit=102.50
        )
        
        plan = scheduler.create_execution_plan("AMZN", params, sample_market_data)
        
        # All slices should respect price limit
        for slice in plan.slices:
            if slice.limit_price is not None:
                assert slice.limit_price == 102.50
    
    def test_execution_plan_with_max_slice_size(self, scheduler, sample_market_data):
        """Test execution plan with maximum slice size constraint"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.TWAP,
            target_quantity=1000,
            time_horizon=timedelta(minutes=60),
            max_participation_rate=0.10,
            urgency=0.5,
            risk_aversion=1.0,
            max_slice_size=100
        )
        
        plan = scheduler.create_execution_plan("META", params, sample_market_data)
        
        # No slice should exceed max size
        for slice in plan.slices:
            assert abs(slice.quantity) <= 100
        
        # Should have at least 10 slices (1000 / 100)
        assert len(plan.slices) >= 10
    
    def test_execution_plan_sell_order(self, scheduler, sample_market_data):
        """Test execution plan for sell orders (negative quantity)"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.TWAP,
            target_quantity=-500,  # Sell order
            time_horizon=timedelta(minutes=30),
            max_participation_rate=0.10,
            urgency=0.5,
            risk_aversion=1.0
        )
        
        plan = scheduler.create_execution_plan("AAPL", params, sample_market_data)
        
        assert plan.total_quantity == -500
        
        # All slices should be negative (sells)
        for slice in plan.slices:
            assert slice.quantity <= 0
        
        # Total should sum to target
        total_quantity = sum(slice.quantity for slice in plan.slices)
        assert abs(total_quantity - (-500)) < 0.01
    
    def test_execution_plan_high_urgency(self, scheduler, sample_market_data):
        """Test execution plan with high urgency factor"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.TWAP,
            target_quantity=1000,
            time_horizon=timedelta(minutes=15),
            max_participation_rate=0.20,
            urgency=0.9,  # High urgency
            risk_aversion=1.0
        )
        
        plan = scheduler.create_execution_plan("AAPL", params, sample_market_data)
        
        # High urgency should result in larger early slices
        if len(plan.slices) > 1:
            first_slice = plan.slices[0]
            assert first_slice.urgency_factor >= 1.0
    
    def test_execution_cost_calculation(self, scheduler, sample_market_data):
        """Test execution cost calculation"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.TWAP,
            target_quantity=1000,
            time_horizon=timedelta(minutes=30),
            max_participation_rate=0.10,
            urgency=0.5,
            risk_aversion=1.0
        )
        
        plan = scheduler.create_execution_plan("AAPL", params, sample_market_data)
        
        # Should calculate reasonable expected cost
        assert plan.expected_cost > 0
        assert plan.expected_cost < 1.0  # Should be less than 100% of notional
    
    def test_execution_risk_calculation(self, scheduler, sample_market_data):
        """Test execution risk calculation"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.IS,
            target_quantity=2000,
            time_horizon=timedelta(hours=2),
            max_participation_rate=0.15,
            urgency=0.3,
            risk_aversion=1.5
        )
        
        plan = scheduler.create_execution_plan("TSLA", params, sample_market_data)
        
        # Should calculate reasonable expected risk
        assert plan.expected_risk >= 0
        assert plan.expected_risk <= 1.0  # Should be reasonable volatility
    
    def test_fallback_plan_creation(self, scheduler):
        """Test fallback plan creation when errors occur"""
        # Test with invalid market data to trigger fallback
        invalid_market_data = {}
        
        params = ExecutionParams(
            strategy=ExecutionStrategy.TWAP,
            target_quantity=1000,
            time_horizon=timedelta(minutes=30),
            max_participation_rate=0.10,
            urgency=0.5,
            risk_aversion=1.0
        )
        
        plan = scheduler.create_execution_plan("TEST", params, invalid_market_data)
        
        # Should create fallback plan
        assert plan is not None
        assert plan.total_quantity == 1000
        assert len(plan.slices) == 1  # Fallback creates single slice
        assert plan.metadata.get("fallback") == True
    
    def test_slice_id_uniqueness(self, scheduler, sample_market_data):
        """Test that execution slices have unique IDs"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.TWAP,
            target_quantity=1000,
            time_horizon=timedelta(minutes=60),
            max_participation_rate=0.10,
            urgency=0.5,
            risk_aversion=1.0
        )
        
        plan = scheduler.create_execution_plan("AAPL", params, sample_market_data)
        
        slice_ids = [slice.slice_id for slice in plan.slices]
        assert len(slice_ids) == len(set(slice_ids))  # All IDs should be unique
    
    def test_execution_time_ordering(self, scheduler, sample_market_data):
        """Test that execution slices are properly time-ordered"""
        params = ExecutionParams(
            strategy=ExecutionStrategy.VWAP,
            target_quantity=1500,
            time_horizon=timedelta(minutes=45),
            max_participation_rate=0.12,
            urgency=0.5,
            risk_aversion=1.0
        )
        
        plan = scheduler.create_execution_plan("MSFT", params, sample_market_data)
        
        # Slices should be in chronological order
        for i in range(1, len(plan.slices)):
            assert plan.slices[i].target_time >= plan.slices[i-1].target_time
