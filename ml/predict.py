import os
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import pickle
from datetime import timedelta
# pyrefly: ignore [missing-import]
from prophet import Prophet

def load_router(model_dir):
    from ml.forecasting import ModelRouter
    registry_path = os.path.join(model_dir, 'model_registry.json')
    return ModelRouter(registry_path)

def forecast_demand(store_id, product_id, horizon_days, model_dir):
    key = f"{store_id}_{product_id}"
    router = load_router(model_dir)
    
    # Load base info
    record = router.registry.get(key)
    if not record:
        return []
        
    m_type = record['model_type']
    metrics = record['metrics']
    reliability = metrics.get('Reliability', 'Low')
    
    demo_data_path = os.path.join(model_dir, 'demo_data.csv')
    df = pd.read_csv(demo_data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to specific store/product
    df_sp = df[(df['store_id'] == store_id) & (df['product_id'] == product_id)].copy()
    if len(df_sp) == 0:
        return []
        
    df_sp = df_sp.sort_values('date').reset_index(drop=True)
    last_row = df_sp.iloc[-1].copy()
    current_date = pd.to_datetime(last_row['date'])
    
    forecasts = []
    
    if m_type == 'xgboost':
        # XGBoost requires iterative autoregressive logic
        current_features = last_row.to_dict()
        
        for i in range(1, horizon_days + 1):
            future_date = current_date + timedelta(days=i)
            current_features['date'] = future_date
            
            # Reconstruct basic calendar features
            current_features['day_of_week'] = future_date.weekday()
            current_features['day_of_month'] = future_date.day
            current_features['week_of_year'] = future_date.isocalendar().week
            current_features['month'] = future_date.month
            current_features['quarter'] = (future_date.month - 1) // 3 + 1
            current_features['is_weekend'] = 1 if current_features['day_of_week'] >= 5 else 0
            
            pred_df = pd.DataFrame([current_features])
            
            # Predict with fallback
            preds, _ = router.predict_with_fallback(key, pred_df)
            pred = preds[0] if preds is not None else 0
            pred = max(0, int(round(pred)))
            
            forecasts.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'store_id': store_id,
                'product_id': product_id,
                'selected_model': m_type,
                'model_version': record.get('version', 1),
                'predicted_demand': pred,
                'validation_metric': 'WAPE',
                'validation_score': metrics.get('WAPE', 0),
                'reliability': reliability
            })
            
            # Autoregressive shift
            if 'lag_1' in current_features:
                current_features['lag_1'] = pred 
                
    elif m_type == 'prophet' or m_type.startswith('baseline_'):
        future_dates = pd.date_range(start=current_date + timedelta(days=1), periods=horizon_days, freq='D')
        future_df = pd.DataFrame({'date': future_dates})
        
        preds, _ = router.predict_with_fallback(key, future_df)
        
        for i, val in enumerate(preds if preds is not None else np.zeros(horizon_days)):
            pred = max(0, int(round(val)))
            forecasts.append({
                'date': future_dates[i].strftime('%Y-%m-%d'),
                'store_id': store_id,
                'product_id': product_id,
                'selected_model': m_type,
                'model_version': record.get('version', 1),
                'predicted_demand': pred,
                'validation_metric': 'WAPE',
                'validation_score': metrics.get('WAPE', 0),
                'reliability': reliability
            })
            
    return forecasts

def calculate_inventory_recommendation(current_stock, safety_stock, lead_time_days, forecasts):
    """
    Separate logic for reordering based on the generated forecasts.
    """
    total_forecast = sum([f['predicted_demand'] for f in forecasts])
    
    reorder_quantity = max(0, total_forecast + safety_stock - current_stock)
    
    if current_stock <= safety_stock:
        status = "CRITICAL"
    elif current_stock < safety_stock + total_forecast * 0.3:
        status = "REORDER NOW"
    elif current_stock < safety_stock + total_forecast * 0.6:
        status = "LOW STOCK"
    else:
        status = "SAFE"
        
    return {
        'total_forecast_demand': total_forecast,
        'recommended_reorder': reorder_quantity,
        'status': status
    }
