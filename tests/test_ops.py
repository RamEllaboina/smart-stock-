import pytest
import pandas as pd
import numpy as np
import os
import json

from ml.ops import (
    DriftDetector,
    MonitoringConfig,
    PerformanceMonitor,
    RetrainingDecisionEngine,
    ModelRegistryLite,
    OpsHealthReport,
    DriftReport,
    PerformanceReport
)

def test_psi_calculation():
    detector = DriftDetector()
    
    # Same distribution
    ref = pd.Series(np.random.normal(10, 2, 1000))
    prod = pd.Series(np.random.normal(10, 2, 1000))
    
    psi_same = detector.calculate_psi(ref, prod)
    assert psi_same < 0.10, "PSI for identical distributions should be low"
    
    # Shifted distribution
    prod_shifted = pd.Series(np.random.normal(15, 2, 1000))
    psi_shifted = detector.calculate_psi(ref, prod_shifted)
    assert psi_shifted > 0.25, "PSI for shifted distributions should be high"
    
def test_performance_monitor():
    config = MonitoringConfig(max_wape=0.15)
    monitor = PerformanceMonitor(config)
    
    y_true = pd.Series([100, 200, 300])
    y_pred = pd.Series([110, 190, 290])
    
    report = monitor.evaluate(y_true, y_pred, ref_wape=0.05)
    
    assert report.mae == 10.0
    # WAPE = 30 / 600 = 0.05
    assert report.wape == 0.05
    assert report.status == "HEALTHY", "Error should be within limits"
    
    # Degraded prediction
    y_pred_bad = pd.Series([50, 100, 150])
    report_bad = monitor.evaluate(y_true, y_pred_bad, ref_wape=0.02)
    assert report_bad.status == "PERFORMANCE_DEGRADATION", "Should flag significant degradation vs reference"

def test_retraining_decision():
    engine = RetrainingDecisionEngine()
    
    healthy = OpsHealthReport(
        production_model="v1",
        model_status="HEALTHY",
        data_quality_score=95,
        drift=DriftReport(total_features=10, features_with_drift=0, percentage_drifted=0, highest_drift_feature="none", prediction_drift_status="LOW", feature_details=[]),
        performance=PerformanceReport(wape=0.05, mae=10, rmse=15, bias=0, residual_mean=0, residual_std=5, status="HEALTHY", degradation=0),
        retraining_decision="PENDING",
        problem_products=0,
        last_training_date="today"
    )
    
    assert engine.decide(healthy) == "NO_ACTION"
    
    # Drifted
    drifted = healthy.model_copy(deep=True)
    drifted.drift.features_with_drift = 5
    assert engine.decide(drifted) == "MONITOR"
    
    # Degraded
    degraded = healthy.model_copy(deep=True)
    degraded.performance.status = "PERFORMANCE_DEGRADATION"
    assert engine.decide(degraded) == "RETRAIN"
    
    # Degraded AND Drifted
    urgent = healthy.model_copy(deep=True)
    urgent.performance.status = "PERFORMANCE_DEGRADATION"
    urgent.drift.features_with_drift = 5
    assert engine.decide(urgent) == "URGENT_RETRAIN"

def test_model_registry_rollback(tmp_path):
    registry = ModelRegistryLite(str(tmp_path))
    
    registry.register_model("s1", "p1", "v1", {"description": "first"})
    assert registry.get_production_model("s1", "p1")["version_id"] == "v1"
    
    registry.register_model("s1", "p1", "v2", {"description": "second"})
    assert registry.get_production_model("s1", "p1")["version_id"] == "v2"
    
    # Check v1 is archived
    assert registry.data["models"]["v1"]["status"] == "ARCHIVED"
    
    success = registry.rollback("s1", "p1")
    assert success is True
    assert registry.get_production_model("s1", "p1")["version_id"] == "v1"
    assert registry.data["models"]["v2"]["status"] == "ROLLED_BACK"
