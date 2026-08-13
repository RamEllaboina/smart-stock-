from pydantic import BaseModel
from typing import Optional, List

class AnomalyConfig(BaseModel):
    iqr_multiplier: float = 1.5
    z_score_threshold: float = 3.0
    rolling_window: int = 7
    rolling_z_threshold: float = 3.0
    enable_seasonal_detection: bool = True
    enable_price_detection: bool = True
    enable_inventory_detection: bool = True
    
    # Context multipliers: e.g. a higher threshold during promotions
    promo_threshold_multiplier: float = 2.0
    weekend_threshold_multiplier: float = 1.2

class AnomalyRecord(BaseModel):
    product_id: str
    store_id: Optional[str] = "STORE_01"
    date: str
    anomaly: bool = True
    anomaly_type: str
    severity: str
    score: float
    reason: str
    original_value: float

class QualityScoreList(BaseModel):
    product_id: str
    store_id: str
    score: float

class AnomalyReport(BaseModel):
    status: str
    total_records: int
    anomalies: int
    critical_anomalies: int
    data_quality_score: float
    product_scores: List[QualityScoreList]
    results: List[AnomalyRecord]
