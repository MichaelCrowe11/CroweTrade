"""Simulation Engine Module"""

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class SimulationConfig:
    scenario_name: str
    num_paths: int
    time_horizon: int
    market_conditions: dict[str, Any]
    seed: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "num_paths": self.num_paths,
            "time_horizon": self.time_horizon,
            "market_conditions": self.market_conditions,
            "seed": self.seed
        }


@dataclass
class SimulationResult:
    scenario_name: str
    paths: list[dict[str, Any]]
    statistics: dict[str, float]
    confidence_intervals: dict[str, tuple]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "paths": self.paths,
            "statistics": self.statistics,
            "confidence_intervals": self.confidence_intervals
        }


class SimulationEngine:
    def __init__(self):
        self.models = {}
        self.results = []
        
    def register_model(self, name: str, model: Any) -> None:
        self.models[name] = model
        
    def run_simulation(self, config: SimulationConfig, model_name: str) -> SimulationResult:
        if config.seed:
            random.seed(config.seed)
            
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
            
        model = self.models[model_name]
        paths = []
        
        for i in range(config.num_paths):
            path = self._simulate_path(model, config)
            paths.append(path)
        
        statistics = self._calculate_statistics(paths)
        confidence_intervals = self._calculate_confidence_intervals(paths)
        
        result = SimulationResult(
            scenario_name=config.scenario_name,
            paths=paths,
            statistics=statistics,
            confidence_intervals=confidence_intervals
        )
        
        self.results.append(result)
        return result
    
    def _simulate_path(self, model: Any, config: SimulationConfig) -> dict[str, Any]:
        path = {"timestamps": [], "values": []}
        current_time = datetime.utcnow()
        
        for t in range(config.time_horizon):
            value = model.simulate_step(t, config.market_conditions)
            path["timestamps"].append(current_time.isoformat())
            path["values"].append(value)
            
        return path
    
    def _calculate_statistics(self, paths: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0
        }
    
    def _calculate_confidence_intervals(self, paths: list[dict[str, Any]]) -> dict[str, tuple]:
        return {
            "95%": (0.0, 0.0),
            "99%": (0.0, 0.0)
        }