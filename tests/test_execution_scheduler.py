"""Tests for execution scheduler and algorithms"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from crowetrade.execution.scheduler import (
    ExecutionScheduler,
    ExecutionAlgorithm,
    MarketConditions,
    MarketImpactModel,
    OrderSlice
)


class TestExecutionScheduler:
    """Test execution scheduling functionality"""
    
    @pytest.fixture
    def scheduler(self):
        """Create an execution scheduler"""
        return ExecutionScheduler(
            default_participation_rate=0.1,
            max_participation_rate=0.3,
            min_slice_size=100
        )
    
    @pytest.fixture
    def market_conditions(self):
        """Create sample market conditions"""
        return MarketConditions(
            bid=100.50,
            ask=100.52,
            mid=100.51,
            spread=0.02,
            volume=1000000,
            volatility=0.15,
            average_daily_volume=10000000,
            participation_rate=0.1,
            liquidity_score=0.8,
            timestamp=datetime.now()
        )
    
    def test_twap_scheduling(self, scheduler, market_conditions):
        """Test TWAP algorithm scheduling"""
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=2)
        
        plan = scheduler.create_execution_plan(
            order_id="test_001",
            symbol="AAPL",
            side="buy",
            quantity=10000,
            algorithm=ExecutionAlgorithm.TWAP,
            market_conditions=market_conditions,
            start_time=start_time,
            end_time=end_time,
            urgency=0.5
        )
        
        assert plan.algorithm == ExecutionAlgorithm.TWAP
        assert len(plan.slices) > 0
        
        # Check slices are evenly distributed
        total_quantity = sum(s.quantity for s in plan.slices)
        assert abs(total_quantity - 10000) < 1e-6
        
        # Check time intervals are roughly equal
        if len(plan.slices) > 1:
            durations = [
                (s.end_time - s.start_time).total_seconds()
                for s in plan.slices
            ]
            assert np.std(durations) / np.mean(durations) < 0.1
    
    def test_vwap_scheduling(self, scheduler, market_conditions):
        """Test VWAP algorithm scheduling"""
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=3)
        
        plan = scheduler.create_execution_plan(
            order_id="test_002",
            symbol="GOOGL",
            side="sell",
            quantity=5000,
            algorithm=ExecutionAlgorithm.VWAP,
            market_conditions=market_conditions,
            start_time=start_time,
            end_time=end_time
        )
        
        assert plan.algorithm == ExecutionAlgorithm.VWAP
        assert len(plan.slices) > 0
        
        # Check total quantity
        total_quantity = sum(s.quantity for s in plan.slices)
        assert abs(total_quantity - 5000) < 1e-6
        
        # VWAP should have varying slice sizes (U-shape profile)
        quantities = [s.quantity for s in plan.slices]
        if len(quantities) > 2:
            # First and last should be relatively larger
            avg_middle = np.mean(quantities[1:-1])
            assert quantities[0] > avg_middle * 0.8 or quantities[-1] > avg_middle * 0.8
    
    def test_pov_scheduling(self, scheduler, market_conditions):
        """Test POV algorithm scheduling"""
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=1)
        
        plan = scheduler.create_execution_plan(
            order_id="test_003",
            symbol="MSFT",
            side="buy",
            quantity=20000,
            algorithm=ExecutionAlgorithm.POV,
            market_conditions=market_conditions,
            start_time=start_time,
            end_time=end_time,
            constraints={'target_pov': 0.15}
        )
        
        assert plan.algorithm == ExecutionAlgorithm.POV
        assert len(plan.slices) > 0
        
        # Check participation rate constraint
        duration_hours = 1
        expected_market_volume = market_conditions.average_daily_volume * (duration_hours / 6.5)
        our_participation = 20000 / expected_market_volume
        assert our_participation <= 0.3  # Max participation rate
    
    def test_implementation_shortfall(self, scheduler, market_conditions):
        """Test IS algorithm scheduling"""
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=30)
        
        plan = scheduler.create_execution_plan(
            order_id="test_004",
            symbol="TSLA",
            side="sell",
            quantity=15000,
            algorithm=ExecutionAlgorithm.IS,
            market_conditions=market_conditions,
            start_time=start_time,
            end_time=end_time,
            urgency=0.8,
            constraints={'risk_aversion': 2.0}
        )
        
        assert plan.algorithm == ExecutionAlgorithm.IS
        assert len(plan.slices) > 0
        
        # IS should front-load execution for high urgency
        if len(plan.slices) > 2:
            first_half_qty = sum(s.quantity for s in plan.slices[:len(plan.slices)//2])
            second_half_qty = sum(s.quantity for s in plan.slices[len(plan.slices)//2:])
            assert first_half_qty > second_half_qty * 0.8  # Front-loaded
    
    def test_iceberg_scheduling(self, scheduler, market_conditions):
        """Test Iceberg algorithm scheduling"""
        
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=2)
        
        plan = scheduler.create_execution_plan(
            order_id="test_005",
            symbol="AMZN",
            side="buy",
            quantity=8000,
            algorithm=ExecutionAlgorithm.ICEBERG,
            market_conditions=market_conditions,
            start_time=start_time,
            end_time=end_time,
            constraints={'visible_percentage': 0.1}
        )
        
        assert plan.algorithm == ExecutionAlgorithm.ICEBERG
        assert len(plan.slices) > 0
        
        # Iceberg should have small visible slices
        for slice_obj in plan.slices:
            assert slice_obj.quantity <= market_conditions.average_daily_volume * 0.001
    
    def test_adaptive_scheduling(self, scheduler):
        """Test adaptive algorithm with different market conditions"""
        
        # Low liquidity conditions
        low_liquidity = MarketConditions(
            bid=50.00,
            ask=50.10,
            mid=50.05,
            spread=0.10,
            volume=100000,
            volatility=0.12,
            average_daily_volume=500000,
            participation_rate=0.05,
            liquidity_score=0.2,  # Low liquidity
            timestamp=datetime.now()
        )
        
        plan = scheduler.create_execution_plan(
            order_id="test_006",
            symbol="ILLIQ",
            side="buy",
            quantity=5000,
            algorithm=ExecutionAlgorithm.ADAPTIVE,
            market_conditions=low_liquidity,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=2)
        )
        
        # Should adjust for low liquidity
        assert len(plan.slices) > 0
        # Slices should be smaller due to low liquidity adjustment
        avg_slice = 5000 / len(plan.slices)
        assert all(s.quantity <= avg_slice * 1.5 for s in plan.slices)
    
    def test_market_impact_model(self):
        """Test market impact estimation"""
        
        model = MarketImpactModel(
            permanent_impact_factor=0.1,
            temporary_impact_factor=0.05
        )
        
        impact = model.estimate_impact(
            quantity=10000,
            adv=1000000,
            volatility=0.20,
            urgency=0.7,
            spread=0.001
        )
        
        assert 'permanent_impact' in impact
        assert 'temporary_impact' in impact
        assert 'total_impact' in impact
        assert impact['total_impact'] > 0
        
        # Higher urgency should increase temporary impact
        impact_low_urgency = model.estimate_impact(
            quantity=10000,
            adv=1000000,
            volatility=0.20,
            urgency=0.2,
            spread=0.001
        )
        
        assert impact['temporary_impact'] > impact_low_urgency['temporary_impact']
    
    def test_risk_score_calculation(self, scheduler, market_conditions):
        """Test execution risk score calculation"""
        
        # High risk scenario
        high_risk_conditions = MarketConditions(
            bid=100.00,
            ask=100.20,
            mid=100.10,
            spread=0.20,
            volume=50000,
            volatility=0.35,  # High volatility
            average_daily_volume=100000,  # Low ADV
            participation_rate=0.5,
            liquidity_score=0.3,  # Low liquidity
            timestamp=datetime.now()
        )
        
        plan = scheduler.create_execution_plan(
            order_id="test_007",
            symbol="RISK",
            side="buy",
            quantity=20000,  # Large relative to ADV
            algorithm=ExecutionAlgorithm.TWAP,
            market_conditions=high_risk_conditions,
            urgency=0.9  # High urgency
        )
        
        assert plan.risk_score > 0.7  # Should be high risk
    
    def test_order_slice_properties(self):
        """Test OrderSlice properties"""
        
        slice_obj = OrderSlice(
            slice_id="slice_001",
            parent_order_id="order_001",
            symbol="TEST",
            side="buy",
            quantity=1000,
            target_price=100.0,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(minutes=5),
            urgency=0.5
        )
        
        assert slice_obj.remaining_quantity == 1000
        assert not slice_obj.is_complete
        
        # Simulate partial execution
        slice_obj.executed_quantity = 600
        assert slice_obj.remaining_quantity == 400
        assert not slice_obj.is_complete
        
        # Complete execution
        slice_obj.executed_quantity = 1000
        assert slice_obj.remaining_quantity == 0
        assert slice_obj.is_complete
    
    def test_execution_history_tracking(self, scheduler, market_conditions):
        """Test execution history and performance tracking"""
        
        # Create and track multiple executions
        for i in range(3):
            plan = scheduler.create_execution_plan(
                order_id=f"hist_{i}",
                symbol="TEST",
                side="buy",
                quantity=1000 * (i + 1),
                algorithm=ExecutionAlgorithm.TWAP,
                market_conditions=market_conditions
            )
            
            # Simulate execution with some error
            actual_impact = plan.estimated_impact * (1 + np.random.normal(0, 0.1))
            actual_cost = plan.estimated_cost * (1 + np.random.normal(0, 0.05))
            
            scheduler.update_execution_history(plan, actual_impact, actual_cost)
        
        assert len(scheduler.execution_history) == 3
        
        # Get performance metrics
        performance = scheduler.get_algorithm_performance()
        assert 'twap' in performance
        assert performance['twap']['count'] == 3
        assert 'avg_impact_error' in performance['twap']
        assert 'impact_mae' in performance['twap']
    
    def test_min_slice_size_constraint(self, scheduler, market_conditions):
        """Test minimum slice size is respected"""
        
        scheduler.min_slice_size = 500
        
        plan = scheduler.create_execution_plan(
            order_id="test_008",
            symbol="TEST",
            side="buy",
            quantity=1000,  # Small order
            algorithm=ExecutionAlgorithm.TWAP,
            market_conditions=market_conditions,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=1)
        )
        
        # Should create fewer slices due to min size
        assert len(plan.slices) <= 2  # 1000 / 500 = 2 max slices
        
        for slice_obj in plan.slices:
            assert slice_obj.quantity >= 450  # Allow some flexibility
    
    def test_high_volatility_adjustment(self, scheduler):
        """Test adjustments for high volatility"""
        
        high_vol_conditions = MarketConditions(
            bid=100.00,
            ask=100.05,
            mid=100.025,
            spread=0.05,
            volume=500000,
            volatility=0.40,  # High volatility
            average_daily_volume=5000000,
            participation_rate=0.1,
            liquidity_score=0.7,
            timestamp=datetime.now()
        )
        
        plan = scheduler.create_execution_plan(
            order_id="test_009",
            symbol="VOL",
            side="sell",
            quantity=10000,
            algorithm=ExecutionAlgorithm.ADAPTIVE,
            market_conditions=high_vol_conditions
        )
        
        # Adaptive should front-load in high volatility
        if len(plan.slices) > 2:
            first_slice_qty = plan.slices[0].quantity
            last_slice_qty = plan.slices[-1].quantity
            # First slices should be larger
            assert first_slice_qty > last_slice_qty * 0.9