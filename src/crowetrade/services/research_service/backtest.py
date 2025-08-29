"""Backtest Engine Module"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class BacktestConfig:
    start_date: datetime
    end_date: datetime
    initial_capital: float
    symbols: list[str]
    strategy_params: dict[str, Any]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "symbols": self.symbols,
            "strategy_params": self.strategy_params
        }


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    trades: list[dict[str, Any]]
    metrics: dict[str, float]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "trades": self.trades,
            "metrics": self.metrics
        }


class BacktestEngine:
    def __init__(self, data_source: Any):
        self.data_source = data_source
        self.results = {}
        
    def run(self, config: BacktestConfig, strategy: Any) -> BacktestResult:
        positions = {}
        trades = []
        capital = config.initial_capital
        
        for symbol in config.symbols:
            market_data = self.data_source.get_historical(
                symbol=symbol,
                start=config.start_date,
                end=config.end_date
            )
            
            for data_point in market_data:
                signal = strategy.generate_signal(data_point, positions)
                if signal:
                    trade = self._execute_trade(signal, data_point, capital)
                    trades.append(trade)
        
        return BacktestResult(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            trades=trades,
            metrics={}
        )
    
    def _execute_trade(self, signal: dict[str, Any], market_data: Any, 
                      capital: float) -> dict[str, Any]:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": signal.get("symbol"),
            "side": signal.get("side"),
            "quantity": signal.get("quantity"),
            "price": market_data.price
        }