from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class MonitoringConfig(BaseModel):
    max_wape: float = 0.15
    max_performance_degradation: float = 0.20
    max_bias: float = 0.10
    psi_low_threshold: float = 0.10
    psi_high_threshold: float = 0.25
    min_new_observations: int = 50
    improvement_margin: float = 0.02 # 2% better to become champion

class FeatureDriftReport(BaseModel):
    feature: str
    drift_score: float
    status: str

class DriftReport(BaseModel):
    total_features: int
    features_with_drift: int
    percentage_drifted: float
    highest_drift_feature: str
    prediction_drift_status: str
    feature_details: List[FeatureDriftReport]

class BiasReport(BaseModel):
    bias: float
    status: str

class PerformanceReport(BaseModel):
    wape: float
    mae: float
    rmse: float
    bias: float
    residual_mean: float
    residual_std: float
    status: str
    degradation: float

class OpsHealthReport(BaseModel):
    production_model: str
    model_status: str
    data_quality_score: float
    drift: DriftReport
    performance: PerformanceReport
    retraining_decision: str
    problem_products: int
    last_training_date: str
    timestamp: str = datetime.now().isoformat()
