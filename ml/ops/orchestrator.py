import pandas as pd
import json
import os
from .schema import OpsHealthReport, MonitoringConfig
from .drift import DriftDetector
from .performance import PerformanceMonitor
from .retraining import RetrainingDecisionEngine
from .registry import ModelRegistryLite

class OpsOrchestrator:
    def __init__(self, model_dir: str):
        self.config = MonitoringConfig()
        self.model_dir = model_dir
        self.registry = ModelRegistryLite(os.path.join(model_dir, 'registry'))
        self.drift_detector = DriftDetector(self.config)
        self.perf_monitor = PerformanceMonitor(self.config)
        self.decision_engine = RetrainingDecisionEngine(self.config)
        
    def generate_health_report(self, store_id: str, product_id: str, production_df: pd.DataFrame, actuals: pd.Series, predictions: pd.Series, quality_score: float) -> OpsHealthReport:
        model_meta = self.registry.get_production_model(store_id, product_id)
        if not model_meta:
            # If no model is registered, fallback to a degraded report saying to train.
            return OpsHealthReport(
                production_model="None",
                model_status="ARCHIVED/MISSING",
                data_quality_score=quality_score,
                drift=self.drift_detector.evaluate_features(pd.DataFrame(), production_df, []),
                performance=self.perf_monitor.evaluate(actuals, predictions, 0.0) or self.perf_monitor.evaluate(pd.Series([0]), pd.Series([0]), 0.0),
                retraining_decision="URGENT_RETRAIN",
                problem_products=1,
                last_training_date="Never"
            )
            
        # Parse references
        features = model_meta.get("features", [])
        ref_wape = model_meta.get("wape", 0.0)
        ref_df_path = os.path.join(self.model_dir, f'ref_data_{store_id}_{product_id}.csv')
        
        if os.path.exists(ref_df_path):
            ref_df = pd.read_csv(ref_df_path)
            drift_report = self.drift_detector.evaluate_features(ref_df, production_df, features)
        else:
            drift_report = self.drift_detector.evaluate_features(production_df, production_df, [])
            
        perf_report = self.perf_monitor.evaluate(actuals, predictions, ref_wape)
        
        # Assemble report
        report = OpsHealthReport(
            production_model=model_meta.get("version_id", "Unknown"),
            model_status="HEALTHY" if perf_report.status == "HEALTHY" else "DEGRADED",
            data_quality_score=quality_score,
            drift=drift_report,
            performance=perf_report,
            retraining_decision="PENDING",
            problem_products=0, # Computed higher up usually
            last_training_date=model_meta.get("registration_date", "Unknown")
        )
        
        report.retraining_decision = self.decision_engine.decide(report)
        return report
        
    def save_report(self, report: OpsHealthReport, store_id: str, product_id: str):
        path = os.path.join(self.model_dir, 'ops_reports')
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f'report_{store_id}_{product_id}.json'), 'w') as f:
            f.write(report.model_dump_json(indent=4))
