from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from datetime import datetime
from typing import List, Optional, Dict, Any

class ForecastRequest(BaseModel):
    dataset_id: str
    fh: List[int]
    correlation_id: str
    frequency: Optional[str] = None

    @field_validator('fh')
    @classmethod
    def fh_must_be_positive(cls, v: List[int]) -> List[int]:
        if any(h <= 0 for h in v):
            raise ValueError('all horizon values must be positive integers')
        return v

class ForecastResponse(BaseModel):
    dataset_id: str
    predictions: List[float]
    prediction_intervals: Optional[Dict[str, List[float]]] = None
    model_version: str
    model_class: str
    model_status: str
    drift_score: Optional[float] = None
    drift_method: Optional[str] = None
    warning: Optional[str] = None
    llm_rationale: Optional[str] = None
    cache_hit: bool
    correlation_id: str

class RetrainJob(BaseModel):
    dataset_id: str
    reason: str
    triggered_at: datetime

class DataProfile(BaseModel):
    # New architecture fields
    dataset_id: str
    n_observations: Optional[int] = None
    narrative: str = ""
    stationarity: Dict[str, Any] = Field(default_factory=dict)
    seasonality: Dict[str, Any] = Field(default_factory=dict)
    structural_break: Dict[str, Any] = Field(default_factory=dict)
    complexity_budget: Dict[str, Any] = Field(default_factory=dict)
    dataset_history: Dict[str, Any] = Field(default_factory=dict)
    training_costs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Legacy fields kept for backward compatibility
    length: Optional[int] = None
    frequency: Optional[str] = None
    has_seasonality: Optional[bool] = None
    is_stationary: Optional[bool] = None
    missing_rate: float = 0.0
    variance: float = 0.0

    model_config = ConfigDict(extra="allow")

    @model_validator(mode="after")
    def _hydrate_legacy_and_new_fields(self):
        if self.n_observations is None and self.length is not None:
            self.n_observations = self.length
        if self.length is None and self.n_observations is not None:
            self.length = self.n_observations

        if self.has_seasonality is None:
            self.has_seasonality = bool(self.seasonality.get("seasonality_class") not in (None, "none"))
        if self.is_stationary is None:
            self.is_stationary = bool(self.stationarity.get("is_stationary", False))

        return self

    def to_natural_language(self) -> str:
        observations = self.n_observations if self.n_observations is not None else self.length
        return (
            f"Time series with {observations} observations, freq={self.frequency}, "
            f"missing rate={self.missing_rate:.1%}"
        )