import os
import pandas as pd
import numpy as np
import xgboost as xgb
import json
import pickle
from datetime import timedelta
from prophet import Prophet

def load_router(model_dir):
    router_path = os.path.join(model_dir, 'router.json')
    if not os.path.exists(router_path):
        return {}
    with open(router_path, 'r') as f:
        return json.load(f)
        
def load_features_list(model_dir):
    features_path = os.path.join(model_dir, 'features.json')
    with open(features_path, 'r') as f:
        return json.load(f)

def forecast_demand(store_id, product_id, horizon_days, model_dir):
    """
    Generate demand forecast using the routed best model for the product.
    """
    key = f"{store_id}_{product_id}"
    router = load_router(model_dir)
    
    if key not in router:
        return []
        
    best_info = router[key]
    best_model = best_info['best_model']
    metrics = best_info['metrics'][best_model]
    
    demo_data_path = os.path.join(model_dir, 'demo_data.csv')
    df = pd.read_csv(demo_data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter to specific store/product
    df_sp = df[(df['store_id'] == store_id) & (df['product_id'] == product_id)].copy()
    if len(df_sp) == 0:
        return []
        
    df_sp = df_sp.sort_values('date').reset_index(drop=True)
    last_row = df_sp.iloc[-1].copy()
    current_date = last_row['date']
    
    forecasts = []
    
    if best_model == 'xgboost':
        features = load_features_list(model_dir)
        m_xgb = xgb.XGBRegressor()
        m_xgb.load_model(os.path.join(model_dir, f'xgb_{key}.json'))
        
        current_features = last_row[features].copy()
        
        for i in range(1, horizon_days + 1):
            future_date = current_date + timedelta(days=i)
            
            current_features['day_of_week'] = future_date.weekday()
            current_features['day_of_month'] = future_date.day
            current_features['week_of_year'] = future_date.isocalendar().week
            current_features['month'] = future_date.month
            current_features['quarter'] = (future_date.month - 1) // 3 + 1
            current_features['is_weekend'] = 1 if current_features['day_of_week'] >= 5 else 0
            
            X = pd.DataFrame([current_features])
            pred = m_xgb.predict(X)[0]
            pred = max(0, int(round(pred)))
            
            forecasts.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'store_id': store_id,
                'product_id': product_id,
                'selected_model': 'XGBoost',
                'predicted_demand': pred,
                'MAE': metrics['MAE']
            })
            
            if 'lag_1' in current_features:
                current_features['lag_1'] = pred 
                
    elif best_model == 'prophet':
        with open(os.path.join(model_dir, f'prophet_{key}.pkl'), 'rb') as f:
            m_prophet = pickle.load(f)
            
        future_dates = pd.date_range(start=current_date + timedelta(days=1), periods=horizon_days, freq='D')
        future_df = pd.DataFrame({'ds': future_dates})
        
        prophet_forecast = m_prophet.predict(future_df)
        
        for i, row in prophet_forecast.iterrows():
            pred = max(0, int(round(row['yhat'])))
            forecasts.append({
                'date': row['ds'].strftime('%Y-%m-%d'),
                'store_id': store_id,
                'product_id': product_id,
                'selected_model': 'Prophet',
                'predicted_demand': pred,
                'MAE': metrics['MAE']
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
