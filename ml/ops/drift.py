import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict

from .schema import MonitoringConfig, FeatureDriftReport, DriftReport

class DriftDetector:
    def __init__(self, config: MonitoringConfig = None):
        self.config = config or MonitoringConfig()

    def calculate_psi(self, expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) between two arrays using quantiles.
        """
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
            
        breakpoints = np.arange(0, buckets + 1) / buckets * 100
        quantiles = np.percentile(expected, breakpoints)
        
        # Ensure unique quantiles to form valid bins
        quantiles = np.unique(quantiles)
        if len(quantiles) < 2:
            return 0.0
            
        def get_percentages(data):
            counts, _ = np.histogram(data, bins=quantiles)
            # Add small epsilon to avoid divide by zero
            counts = np.maximum(counts, 1) 
            return counts / sum(counts)
            
        expected_pct = get_percentages(expected)
        actual_pct = get_percentages(actual)
        
        psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        return float(np.sum(psi_values))

    def evaluate_features(self, reference_df: pd.DataFrame, production_df: pd.DataFrame, numerical_features: List[str]) -> DriftReport:
        reports = []
        drift_count = 0
        max_drift = 0.0
        max_feature = "None"
        
        for feature in numerical_features:
            if feature not in reference_df.columns or feature not in production_df.columns:
                continue
                
            ref = reference_df[feature].dropna()
            prod = production_df[feature].dropna()
            
            # Simple PSI for numerical features
            psi = self.calculate_psi(ref, prod)
            
            if psi < self.config.psi_low_threshold:
                status = "LOW"
            elif psi < self.config.psi_high_threshold:
                status = "MODERATE"
                drift_count += 1
            else:
                status = "HIGH"
                drift_count += 1
                
            if psi > max_drift:
                max_drift = psi
                max_feature = feature
                
            reports.append(FeatureDriftReport(feature=feature, drift_score=round(psi, 3), status=status))
            
        return DriftReport(
            total_features=len(reports),
            features_with_drift=drift_count,
            percentage_drifted=round((drift_count / max(1, len(reports))) * 100, 2),
            highest_drift_feature=max_feature,
            prediction_drift_status="UNKNOWN", # calculated in another step usually on predictions
            feature_details=reports
        )
