import pandas as pd
import numpy as np
from typing import Dict, List
from .schema import MonitoringConfig, PerformanceReport

class PerformanceMonitor:
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series, ref_wape: float = None) -> PerformanceReport:
        if len(y_true) == 0:
            return None
            
        y_true = np.maximum(y_true, 0)
        y_pred = np.maximum(y_pred, 0)
        
        residuals = y_true - y_pred
        bias = float(np.mean(residuals))
        
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))
        
        # WAPE calculation
        sum_actual = np.sum(y_true)
        wape = float(np.sum(np.abs(residuals)) / sum_actual) if sum_actual > 0 else mae
        
        res_mean = float(np.mean(residuals))
        res_std = float(np.std(residuals))
        
        degradation = 0.0
        status = "HEALTHY"
        
        if ref_wape is not None and ref_wape > 0:
            degradation = (wape - ref_wape) / ref_wape
            if degradation > self.config.max_performance_degradation:
                status = "PERFORMANCE_DEGRADATION"
                
        if status == "HEALTHY":
            if wape > self.config.max_wape:
                status = "HIGH_ERROR"
            elif abs(bias) / (np.mean(y_true)+1e-5) > self.config.max_bias:
                status = "HIGH_BIAS"

        return PerformanceReport(
            wape=round(wape, 4),
            mae=round(mae, 4),
            rmse=round(rmse, 4),
            bias=round(bias, 4),
            residual_mean=round(res_mean, 4),
            residual_std=round(res_std, 4),
            status=status,
            degradation=round(degradation, 4)
        )
