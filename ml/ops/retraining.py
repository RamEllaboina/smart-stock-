from .schema import OpsHealthReport, MonitoringConfig
from typing import Dict

class RetrainingDecisionEngine:
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()
        
    def decide(self, report: OpsHealthReport) -> str:
        """
        Inputs: Data Drift, Performance, Anomaly Rate (from Data Quality)
        Output: NO_ACTION, MONITOR, INVESTIGATE, RETRAIN, URGENT_RETRAIN
        """
        
        if report.data_quality_score < 70:
            return "INVESTIGATE" # Data is corrupted, retraining might be dangerous natively
            
        perf = report.performance
        drift = report.drift
        
        is_perf_degraded = perf.status in ["PERFORMANCE_DEGRADATION", "HIGH_ERROR"]
        is_high_bias = perf.status == "HIGH_BIAS"
        is_drifted = drift.features_with_drift > (drift.total_features * 0.3) if drift.total_features > 0 else False
        
        if is_perf_degraded and is_drifted:
            return "URGENT_RETRAIN"
            
        if is_perf_degraded:
            return "RETRAIN"
            
        if is_high_bias or is_drifted:
            return "MONITOR"
            
        return "NO_ACTION"
        
    def promote_candidate(self, ref_wape: float, cand_wape: float) -> bool:
        """
        Champion / Challenger evaluation
        """
        # Lower WAPE is better. Cand WAPE must be significantly lower than Ref WAPE
        if ref_wape is None or ref_wape == 0.0:
            return True
            
        improvement = (ref_wape - cand_wape) / ref_wape
        return improvement >= self.config.improvement_margin
