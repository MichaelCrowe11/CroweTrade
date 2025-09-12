"""Model Management Package

This package provides model registry and A/B testing capabilities:
- ModelRegistry: Centralized model versioning and deployment
- ABTestEngine: Statistical model comparison and optimization
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

# Model Registry imports
from .registry import (
    ModelRegistry,
    ModelMetadata, 
    ModelArtifacts,
    ModelStatus,
    ModelType,
    create_model_metadata
)

# A/B Testing imports
from .ab_testing import (
    ABTestEngine,
    ABTestConfig,
    TestArm,
    AllocationStrategy,
    TestStatus,
    create_ab_test
)


class ModelType(Enum):
    LINEAR = "linear"
    TREE = "tree"
    ENSEMBLE = "ensemble"
    NEURAL = "neural"
    FACTOR = "factor"
    REGIME = "regime"
    CAUSAL = "causal"


class ModelStatus(Enum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


@dataclass
class ModelCard:
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus
    
    author: str
    pod: str  # which quant pod owns this
    created_at: datetime
    updated_at: datetime
    
    description: str
    features: list[str]
    target: str
    horizon: str
    
    metrics: dict[str, float] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    
    validation_results: dict[str, Any] = field(default_factory=dict)
    backtest_results: dict[str, Any] = field(default_factory=dict)
    
    deployment_config: dict[str, Any] = field(default_factory=dict)
    monitoring_config: dict[str, Any] = field(default_factory=dict)
    
    signatures: list[str] = field(default_factory=list)
    approvals: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Prediction:
    instrument: str
    horizon: str
    timestamp: datetime
    
    mu: float  # expected return
    sigma: float  # uncertainty/volatility
    prob_positive: float  # probability of positive return
    confidence: float  # model confidence
    
    model_id: str
    features_used: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_kelly_fraction(self, risk_free: float = 0.0) -> float:
        if self.sigma <= 0:
            return 0.0
        
        edge = self.mu - risk_free
        if edge <= 0:
            return 0.0
        
        return min(0.25, edge / (self.sigma ** 2))  # capped Kelly
    
    def get_risk_adjusted_size(self, base_size: float,
                               risk_multiplier: float = 0.5) -> float:
        kelly = self.get_kelly_fraction()
        confidence_adj = self.confidence ** 2  # square for conservatism
        return base_size * kelly * risk_multiplier * confidence_adj


class Model(ABC):
    def __init__(self, model_card: ModelCard):
        self.model_card = model_card
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass
    
    def get_feature_importance(self) -> dict[str, float]:
        return {}


class EnsembleModel(Model):
    def __init__(self, model_card: ModelCard, base_models: list[Model]):
        super().__init__(model_card)
        self.base_models = base_models
        self.weights = np.ones(len(base_models)) / len(base_models)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        for model in self.base_models:
            model.fit(X, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([model.predict(X) for model in self.base_models])
        return np.average(predictions, weights=self.weights, axis=0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = np.array([model.predict_proba(X) for model in self.base_models])
        return np.average(probas, weights=self.weights, axis=0)
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray,
                        metric: str = "sharpe") -> None:
        from scipy.optimize import minimize
        
        def objective(weights):
            weights = weights / weights.sum()
            predictions = np.array([m.predict(X_val) for m in self.base_models])
            ensemble_pred = np.average(predictions, weights=weights, axis=0)
            
            if metric == "sharpe":
                returns = ensemble_pred * y_val
                return -np.mean(returns) / (np.std(returns) + 1e-8)
            elif metric == "mse":
                return np.mean((ensemble_pred - y_val) ** 2)
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        result = minimize(
            objective,
            self.weights,
            bounds=[(0, 1)] * len(self.weights),
            constraints={"type": "eq", "fun": lambda w: w.sum() - 1}
        )
        
        if result.success:
            self.weights = result.x / result.x.sum()


class RegimeModel(Model):
    def __init__(self, model_card: ModelCard, n_regimes: int = 2):
        super().__init__(model_card)
        self.n_regimes = n_regimes
        self.regime_models: dict[int, Model] = {}
        self.current_regime = 0
        self.regime_probabilities = np.ones(n_regimes) / n_regimes
    
    def detect_regime(self, X: np.ndarray) -> int:
        return 0
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        regimes = self._cluster_regimes(X, y)
        
        for regime in range(self.n_regimes):
            mask = regimes == regime
            if mask.sum() > 0:
                X_regime = X[mask]
                y_regime = y[mask]
        
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        regime = self.detect_regime(X)
        if regime in self.regime_models:
            return self.regime_models[regime].predict(X)
        return np.zeros(len(X))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        regime = self.detect_regime(X)
        if regime in self.regime_models:
            return self.regime_models[regime].predict_proba(X)
        return np.ones((len(X), 2)) * 0.5
    
    def _cluster_regimes(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.cluster import KMeans
        
        features = np.column_stack([
            np.std(y.reshape(-1, 20), axis=1) if len(y) >= 20 else np.std(y),
            np.mean(X, axis=1) if X.ndim > 1 else X
        ])
        
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42)
        return kmeans.fit_predict(features[:, :2] if features.shape[1] > 2 else features)


class ModelValidator(Protocol):
    def validate(self, model: Model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        ...
    
    def check_drift(self, model: Model, X_new: np.ndarray) -> bool:
        ...