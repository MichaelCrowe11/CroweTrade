"""Data Validation Module"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationRule:
    field: str
    rule_type: str
    params: dict[str, Any]
    
    def validate(self, value: Any) -> bool:
        if self.rule_type == "range":
            min_val = self.params.get("min", float("-inf"))
            max_val = self.params.get("max", float("inf"))
            return min_val <= value <= max_val
        elif self.rule_type == "required":
            return value is not None
        elif self.rule_type == "type":
            expected_type = self.params.get("type")
            return isinstance(value, expected_type)
        return True


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings
        }


class DataValidator:
    def __init__(self, rules: list[ValidationRule]):
        self.rules = rules
        
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        errors = []
        warnings = []
        
        for rule in self.rules:
            field_value = data.get(rule.field)
            
            if not rule.validate(field_value):
                errors.append(f"Validation failed for field '{rule.field}': {rule.rule_type}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )