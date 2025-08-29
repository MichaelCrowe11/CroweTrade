import pytest
import random
import numpy as np
from datetime import UTC, datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from crowetrade.core.contracts import FeatureVector, Signal, TargetPosition, Fill
from crowetrade.live.signal_agent import SignalAgent
from crowetrade.live.portfolio_agent import PortfolioAgent
from crowetrade.live.risk_guard import RiskGuard


@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    trades: List[Fill]
    pnl_series: List[float]
    positions: Dict[str, List[float]]


@dataclass
class MarketSimulator:
    """Deterministic market simulator for backtesting."""
    seed: int = 42
    base_spread: float = 0.0001
    slippage: float = 0.0001
    
    def __post_init__(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
    
    def generate_prices(self, 
                       instrument: str, 
                       start_date: datetime,
                       n_days: int,
                       volatility: float = 0.02) -> Dict[datetime, float]:
        """Generate deterministic price series."""
        prices = {}
        current_price = 100.0
        current_date = start_date
        
        # Deterministic random walk using combined seed
        combined_seed = (self.seed + hash(instrument)) % 2**32
        np.random.seed(combined_seed)
        returns = np.random.normal(0, volatility, n_days)
        
        for i in range(n_days):
            prices[current_date] = current_price
            current_price *= (1 + returns[i])
            current_date += timedelta(days=1)
        
        return prices
    
    def execute_order(self, 
                     instrument: str,
                     qty: float,
                     price: float,
                     timestamp: datetime) -> Fill:
        """Simulate order execution with slippage."""
        # Deterministic slippage based on order size
        slippage_factor = 1 + self.slippage * np.sign(qty)
        exec_price = price * slippage_factor
        
        return Fill(
            instrument=instrument,
            qty=qty,
            price=exec_price,
            ts=timestamp,
            venue="SIM"
        )


class BacktestEngine:
    """Backtesting engine with deterministic behavior."""
    
    def __init__(self, 
                 signal_agent: SignalAgent,
                 portfolio_agent: PortfolioAgent,
                 risk_guard: RiskGuard,
                 simulator: MarketSimulator):
        self.signal_agent = signal_agent
        self.portfolio_agent = portfolio_agent
        self.risk_guard = risk_guard
        self.simulator = simulator
    
    def run_backtest(self,
                    instruments: List[str],
                    start_date: datetime,
                    end_date: datetime,
                    initial_capital: float = 100000) -> BacktestResult:
        """Run deterministic backtest."""
        # Generate price data
        n_days = (end_date - start_date).days
        market_data = {}
        for instrument in instruments:
            market_data[instrument] = self.simulator.generate_prices(
                instrument, start_date, n_days
            )
        
        # Initialize tracking
        capital = initial_capital
        positions = {inst: [] for inst in instruments}
        trades = []
        pnl_series = []
        current_positions = {inst: 0 for inst in instruments}
        
        # Simulate each day
        current_date = start_date
        while current_date <= end_date:
            daily_pnl = 0
            
            # Generate features and signals
            signals = {}
            for instrument in instruments:
                if current_date in market_data[instrument]:
                    price = market_data[instrument][current_date]
                    
                    # Create feature vector (simplified)
                    fv = FeatureVector(
                        instrument=instrument,
                        asof=current_date,
                        horizon="1d",
                        values={"mom": np.random.normal(0, 0.02)},
                        quality={"lag_ms": 50, "coverage": 1.0}
                    )
                    
                    # Generate signal
                    signal = self.signal_agent.infer(fv)
                    if signal:
                        signals[instrument] = signal
            
            # Size positions
            if signals:
                tracking_errors = {inst: 0.02 for inst in signals}
                targets = self.portfolio_agent.size(signals, tracking_errors)
                
                # Execute trades with risk checks
                for instrument, target_qty in targets.items():
                    current_qty = current_positions.get(instrument, 0)
                    trade_qty = target_qty - current_qty
                    
                    if trade_qty != 0:
                        price = market_data[instrument][current_date]
                        exposure = abs(trade_qty * price)
                        var_est = exposure * 0.02
                        
                        # Risk check
                        if self.risk_guard.pretrade_check(exposure, var_est):
                            fill = self.simulator.execute_order(
                                instrument, trade_qty, price, current_date
                            )
                            trades.append(fill)
                            current_positions[instrument] = target_qty
                            
                            # Update PnL
                            trade_pnl = -trade_qty * fill.price
                            daily_pnl += trade_pnl
            
            # Mark to market
            for instrument, qty in current_positions.items():
                if qty != 0 and current_date in market_data[instrument]:
                    price_change = 0  # Simplified
                    if current_date > start_date:
                        prev_date = current_date - timedelta(days=1)
                        if prev_date in market_data[instrument]:
                            price_change = (market_data[instrument][current_date] - 
                                          market_data[instrument][prev_date])
                    daily_pnl += qty * price_change
            
            # Update risk guard
            self.risk_guard.update_pnl(daily_pnl)
            
            # Track results
            pnl_series.append(daily_pnl)
            capital += daily_pnl
            
            for inst in instruments:
                positions[inst].append(current_positions.get(inst, 0))
            
            current_date += timedelta(days=1)
        
        # Calculate metrics
        total_return = (capital - initial_capital) / initial_capital
        returns = np.array(pnl_series) / initial_capital
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Calculate max drawdown
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / initial_capital
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Win rate
        winning_trades = sum(1 for t in trades if t.qty > 0)
        win_rate = winning_trades / len(trades) if trades else 0
        
        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            trades=trades,
            pnl_series=pnl_series,
            positions=positions
        )


class TestDeterministicBacktest:
    """Test deterministic backtesting."""
    
    @pytest.mark.simulation
    def test_backtest_determinism(self):
        """Test that backtests are deterministic with fixed seed."""
        def model(values):
            return values.get("mom", 0), 0.02, 0.6
        
        # Run backtest twice with same seed
        results = []
        for _ in range(2):
            signal_agent = SignalAgent(
                model=model,
                policy_id="test",
                gates={"prob_edge_min": 0.55, "sigma_max": 0.03}
            )
            portfolio_agent = PortfolioAgent(risk_budget=1.0, lambda_temper=0.25)
            risk_guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
            simulator = MarketSimulator(seed=42)
            
            engine = BacktestEngine(
                signal_agent, portfolio_agent, risk_guard, simulator
            )
            
            result = engine.run_backtest(
                instruments=["AAPL", "GOOGL"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 31, tzinfo=UTC),
                initial_capital=100000
            )
            results.append(result)
        
        # Results should be identical
        assert results[0].total_return == results[1].total_return
        assert results[0].sharpe_ratio == results[1].sharpe_ratio
        assert results[0].max_drawdown == results[1].max_drawdown
        assert len(results[0].trades) == len(results[1].trades)
    
    @pytest.mark.simulation
    def test_backtest_different_seeds(self):
        """Test that different seeds produce different results."""
        def model(values):
            return values.get("mom", 0), 0.02, 0.6
        
        results = []
        for seed in [42, 123]:
            signal_agent = SignalAgent(
                model=model,
                policy_id="test",
                gates={"prob_edge_min": 0.55, "sigma_max": 0.03}
            )
            portfolio_agent = PortfolioAgent(risk_budget=1.0, lambda_temper=0.25)
            risk_guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
            simulator = MarketSimulator(seed=seed)
            
            engine = BacktestEngine(
                signal_agent, portfolio_agent, risk_guard, simulator
            )
            
            result = engine.run_backtest(
                instruments=["AAPL"],
                start_date=datetime(2024, 1, 1, tzinfo=UTC),
                end_date=datetime(2024, 1, 31, tzinfo=UTC),
                initial_capital=100000
            )
            results.append(result)
        
        # Results should be different
        assert results[0].total_return != results[1].total_return
    
    @pytest.mark.simulation
    def test_risk_limits_enforced(self):
        """Test that risk limits are enforced during backtest."""
        def model(values):
            return 0.10, 0.50, 0.9  # High risk signal
        
        signal_agent = SignalAgent(
            model=model,
            policy_id="test",
            gates={"prob_edge_min": 0.5, "sigma_max": 1.0}
        )
        portfolio_agent = PortfolioAgent(risk_budget=1.0, lambda_temper=0.25)
        risk_guard = RiskGuard(dd_limit=0.02, var_limit=0.01)  # Tight limits
        simulator = MarketSimulator(seed=42)
        
        engine = BacktestEngine(
            signal_agent, portfolio_agent, risk_guard, simulator
        )
        
        result = engine.run_backtest(
            instruments=["AAPL"],
            start_date=datetime(2024, 1, 1, tzinfo=UTC),
            end_date=datetime(2024, 1, 7, tzinfo=UTC),
            initial_capital=100000
        )
        
        # Should have limited trades due to risk constraints
        assert len(result.trades) < 7  # Less than daily trading
        assert result.max_drawdown >= -0.02  # Respects drawdown limit
    
    @pytest.mark.simulation
    @pytest.mark.slow
    def test_cross_validation(self):
        """Test cross-validation with multiple time periods."""
        def model(values):
            return values.get("mom", 0), 0.02, 0.6
        
        periods = [
            (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 31, tzinfo=UTC)),
            (datetime(2024, 2, 1, tzinfo=UTC), datetime(2024, 2, 29, tzinfo=UTC)),
            (datetime(2024, 3, 1, tzinfo=UTC), datetime(2024, 3, 31, tzinfo=UTC)),
        ]
        
        results = []
        for start_date, end_date in periods:
            signal_agent = SignalAgent(
                model=model,
                policy_id="test",
                gates={"prob_edge_min": 0.55, "sigma_max": 0.03}
            )
            portfolio_agent = PortfolioAgent(risk_budget=1.0, lambda_temper=0.25)
            risk_guard = RiskGuard(dd_limit=0.05, var_limit=0.02)
            simulator = MarketSimulator(seed=42)
            
            engine = BacktestEngine(
                signal_agent, portfolio_agent, risk_guard, simulator
            )
            
            result = engine.run_backtest(
                instruments=["AAPL", "GOOGL", "MSFT"],
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000
            )
            results.append(result)
        
        # Check consistency across periods
        sharpe_ratios = [r.sharpe_ratio for r in results]
        assert all(-2 < sr < 2 for sr in sharpe_ratios)  # Reasonable range
        
        # Average metrics
        avg_return = np.mean([r.total_return for r in results])
        avg_sharpe = np.mean(sharpe_ratios)
        
        assert -0.5 < avg_return < 0.5  # Reasonable return range
        assert -1 < avg_sharpe < 1  # Reasonable Sharpe range