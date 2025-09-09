"""Model Training Module"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class TrainingConfig:
    model_type: str
    hyperparameters: dict[str, Any]
    train_split: float
    validation_split: float
    epochs: int | None = None
    batch_size: int | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "hyperparameters": self.hyperparameters,
            "train_split": self.train_split,
            "validation_split": self.validation_split,
            "epochs": self.epochs,
            "batch_size": self.batch_size
        }


@dataclass
class TrainingResult:
    model_id: str
    model_checksum: str
    metrics: dict[str, float]
    training_time: float
    config: TrainingConfig
    metadata: dict[str, Any] | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_checksum": self.model_checksum,
            "metrics": self.metrics,
            "training_time": self.training_time,
            "config": self.config.to_dict(),
            "metadata": self.metadata
        }


class ModelTrainer:
    def __init__(self, model_registry: Any):
        self.model_registry = model_registry
        self.model_builders = {}
        
    def register_model_builder(self, model_type: str, builder: Any) -> None:
        self.model_builders[model_type] = builder
        
    def train(self, data: list[Any], config: TrainingConfig) -> TrainingResult:
        if config.model_type not in self.model_builders:
            raise ValueError(f"Unknown model type: {config.model_type}")
        
        train_data, val_data, test_data = self._split_data(data, config)
        
        model_builder = self.model_builders[config.model_type]
        model = model_builder.build(config.hyperparameters)
        
        start_time = datetime.utcnow()
        
        model.fit(train_data, validation_data=val_data, 
                 epochs=config.epochs, batch_size=config.batch_size)
        
        training_time = (datetime.utcnow() - start_time).total_seconds()
        
        metrics = self._evaluate_model(model, test_data)
        
        model_id = self._generate_model_id(config)
        model_checksum = self._calculate_checksum(model)
        
        self.model_registry.register(model_id, model, config, model_checksum)
        
        return TrainingResult(
            model_id=model_id,
            model_checksum=model_checksum,
            metrics=metrics,
            training_time=training_time,
            config=config,
            metadata={"trained_at": datetime.utcnow().isoformat()}
        )
    
    def _split_data(self, data: list[Any], config: TrainingConfig) -> tuple[list, list, list]:
        n = len(data)
        train_end = int(n * config.train_split)
        val_end = train_end + int(n * config.validation_split)
        
        return data[:train_end], data[train_end:val_end], data[val_end:]
    
    def _evaluate_model(self, model: Any, test_data: list[Any]) -> dict[str, float]:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
    
    def _generate_model_id(self, config: TrainingConfig) -> str:
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return f"{config.model_type}_{hashlib.md5(config_str.encode()).hexdigest()[:8]}"
    
    def _calculate_checksum(self, model: Any) -> str:
        return hashlib.sha256(str(model).encode()).hexdigest()