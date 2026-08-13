import pandas as pd
import numpy as np
import os
import sys

from ml.ops.orchestrator import OpsOrchestrator

def mock_ops_reports():
    model_dir = os.path.join("ml", "model")
    data_path = os.path.join(model_dir, "demo_data.csv")
    
    if not os.path.exists(data_path):
        print("Data not found")
        return
        
    df = pd.read_csv(data_path)
    stores = df['store_id'].unique()
    products = df['product_id'].unique()
    
    orchestrator = OpsOrchestrator(model_dir)
    
    # Generate mock production data to run drift against
    print(f"Generating mock OpsHealthReports...")
    for s in stores:
        for p in products:
            # We don't need real accurate data, just enough to pass into orchestrator mapping
            mock_prod_df = pd.DataFrame({'sales': np.random.normal(50, 10, 30)})
            actuals = pd.Series(np.random.normal(50, 10, 30))
            
            # Inject some random error so the WAPE/MAE looks realistic
            # 5% chance of severe degradation
            if np.random.random() < 0.05:
                # Terrible predictions
                predictions = pd.Series(np.random.normal(150, 30, 30))
                qs = 60.0
            else:
                # Decent predictions
                predictions = actuals + np.random.normal(0, 5, 30)
                qs = 95.0
                
            report = orchestrator.generate_health_report(
                store_id=s,
                product_id=p,
                production_df=mock_prod_df,
                actuals=actuals,
                predictions=predictions,
                quality_score=qs
            )
            
            orchestrator.save_report(report, s, p)
            
    print("Ops reports successfully generated.")

if __name__ == "__main__":
    mock_ops_reports()
