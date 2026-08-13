import os
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import sys
from prophet import Prophet
import warnings

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocessing import preprocess_pipeline
from ml.features import feature_engineering_pipeline, time_series_split
from ml.evaluate import evaluate_metrics

def prepare_prophet_data(df):
    prophet_df = df[['date', 'sales']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def train_models_and_route(data_path, model_dir):
    print("Loading raw data for validation...")
    df_raw = pd.read_csv(data_path)
    
    print("Running Production Data Validation...")
    from validation.validator import DataValidator
    validator = DataValidator(df_raw, min_history_length=30)
    report = validator.validate()
    print("\n" + str(report))
    
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'validation_report.json'), 'w') as f:
        f.write(report.to_json())
        
    if report.status == "FAIL":
        print("\nData validation FAILED. Stopping pipeline.")
        sys.exit(1)
        
    print("\nStarting Production Anomaly Detection & Quality Monitoring...")
    # Use the structurally validated and properly renamed dataset from the report
    df_scan = report.df.copy()
        
    from ml.anomaly import AnomalyDetector, DataQualityScorer
    detector = AnomalyDetector()
    df_anomalies = detector.process(df_scan)
    
    scorer = DataQualityScorer()
    anomaly_report = scorer.evaluate(df_anomalies)
    
    # Save Report
    with open(os.path.join(model_dir, 'anomaly_report.json'), 'w') as f:
        f.write(anomaly_report.model_dump_json(indent=4))
        
    # Print lightweight console report
    print("Smart Stock Anomaly Report")
    print("===========================")
    print(f"Total Records: {anomaly_report.total_records}")
    print(f"Anomalies Detected: {anomaly_report.anomalies}")
    print(f"Critical Anomalies: {anomaly_report.critical_anomalies}")
    print(f"Data Quality Score: {anomaly_report.data_quality_score}/100")
    print(f"Status: {anomaly_report.status}\n")
        
    print("\nProceeding with Preprocessing...")
    df = preprocess_pipeline(data_path)
    print("Feature Engineering for XGBoost...")
    df_feat = feature_engineering_pipeline(df)
    
    drop_cols = ['date', 'sales', 'is_outlier', 'safety_stock', 'lead_time_days', 'price', 
                 'store_id', 'product_id', 'category']
    numeric_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_cols if c not in drop_cols]
    target = 'sales'
    
    print("Splitting data chronologically...")
    train_df, val_df, test_df = time_series_split(df_feat)
    
    from ml.forecasting import ModelRouter
    
    # Initialize Model Router
    os.makedirs(model_dir, exist_ok=True)
    registry_path = os.path.join(model_dir, 'model_registry.json')
    router = ModelRouter(registry_path)
    
    # Group by store_id and product_id
    group_keys = train_df[['store_id', 'product_id']].drop_duplicates().values
    
    print(f"Training and Evaluating Candidates for {len(group_keys)} series...")
    
    for store_id, product_id in group_keys:
        p_train = train_df[(train_df['store_id'] == store_id) & (train_df['product_id'] == product_id)]
        p_val = val_df[(val_df['store_id'] == store_id) & (val_df['product_id'] == product_id)]
        
        router.train_and_select(p_train, p_val, store_id, product_id, model_dir)
        
    # Pre-calculate demo data for dashboards (last 60 days)
    max_date = df_feat['date'].max()
    demo_df = df_feat[df_feat['date'] >= (max_date - pd.Timedelta(days=60))]
    demo_df.to_csv(os.path.join(model_dir, 'demo_data.csv'), index=False)
    
    df_clean = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=60))]
    df_clean.to_csv(os.path.join(model_dir, 'demo_raw_data.csv'), index=False)
    
    print("\nStarting MLOps Analysis...")
    try:
        from ml.ops import OpsOrchestrator
        orchestrator = OpsOrchestrator(model_dir)
        for store_id, product_id in group_keys:
            key = f"{store_id}_{product_id}"
            p_val = val_df[(val_df['store_id'] == store_id) & (val_df['product_id'] == product_id)]
            
            if len(p_val) == 0:
                continue
                
            y_true = p_val['sales']
            y_pred, _ = router.predict_with_fallback(key, p_val)
            if y_pred is None:
                continue
            
            q_score = anomaly_report.data_quality_score if anomaly_report else 100.0
            
            health = orchestrator.generate_health_report(
                store_id, product_id, 
                production_df=p_val, 
                actuals=y_true, 
                predictions=y_pred, 
                quality_score=q_score
            )
            orchestrator.save_report(health, store_id, product_id)
        print("MLOps Health reports generated for all available targets.")
    except Exception as e:
        print(f"MLOps analysis fail: {e}")
    
    print("\n--- Training Complete ---")
    print(f"Multi-model registry saved to {registry_path}")

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'sales_data.csv')
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    train_models_and_route(data_path, model_dir)
