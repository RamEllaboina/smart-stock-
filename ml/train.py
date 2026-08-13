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
from ml.features import feature_engineering_pipeline
from ml.evaluate import evaluate_metrics

def time_series_split(df, train_prop=0.7, val_prop=0.15):
    """
    Chronological split for time series
    """
    df = df.sort_values('date')
    dates = df['date'].unique()
    
    n = len(dates)
    train_end = int(n * train_prop)
    val_end = int(n * (train_prop + val_prop))
    
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    
    train = df[df['date'].isin(train_dates)]
    val = df[df['date'].isin(val_dates)]
    test = df[df['date'].isin(test_dates)]
    
    return train, val, test

def prepare_prophet_data(df):
    prophet_df = df[['date', 'sales']].copy()
    prophet_df.columns = ['ds', 'y']
    return prophet_df

def train_models_and_route(data_path, model_dir):
    print("Loading and Preprocessing data...")
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
    
    # Store trained models and router mapping
    os.makedirs(model_dir, exist_ok=True)
    router_mapping = {}
    aggregated_metrics = {}
    
    # Group by store_id and product_id
    group_keys = train_df[['store_id', 'product_id']].drop_duplicates().values
    
    print(f"Training and Evaluating Models for {len(group_keys)} products...")
    
    for store_id, product_id in group_keys:
        print(f"--- Processing Store: {store_id} | Product: {product_id} ---")
        
        # Filter data for this product
        p_train = train_df[(train_df['store_id'] == store_id) & (train_df['product_id'] == product_id)]
        p_val = val_df[(val_df['store_id'] == store_id) & (val_df['product_id'] == product_id)]
        p_test = test_df[(test_df['store_id'] == store_id) & (test_df['product_id'] == product_id)]
        
        if len(p_train) < 30 or len(p_val) < 5:
            print(f"Skipping {product_id} due to insufficient data.")
            continue
            
        # --- 1. Train & Evaluate Prophet ---
        prophet_train = prepare_prophet_data(p_train)
        prophet_val = prepare_prophet_data(p_val)
        
        m_prophet = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
        try:
            m_prophet.fit(prophet_train)
            future = m_prophet.make_future_dataframe(periods=len(p_val), freq='D')
            forecast = m_prophet.predict(future)
            # Align predictions with validation set dates
            val_dates = p_val['date'].values
            prophet_preds = forecast[forecast['ds'].isin(val_dates)]['yhat'].values
            
            # Ensure no negative predictions
            prophet_preds = np.maximum(prophet_preds, 0)
            
            # If for some reason lengths don't match, fallback
            if len(prophet_preds) != len(p_val):
                prophet_preds = np.zeros(len(p_val))
                
        except Exception as e:
            print(f"Prophet failed for {product_id}: {e}")
            prophet_preds = np.zeros(len(p_val))
            
        prophet_metrics = evaluate_metrics(p_val[target], prophet_preds)
        
        # --- 2. Train & Evaluate XGBoost ---
        X_train, y_train = p_train[features], p_train[target]
        X_val, y_val = p_val[features], p_val[target]
        
        m_xgb = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        m_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        xgb_preds = m_xgb.predict(X_val)
        xgb_preds = np.maximum(xgb_preds, 0)
        
        xgb_metrics = evaluate_metrics(y_val, xgb_preds)
        
        # --- 3. Model Router (Select Best Model) ---
        # We'll use MAE as the primary metric for selection
        best_model = "xgboost"
        best_mae = xgb_metrics['MAE']
        
        if prophet_metrics['MAE'] < xgb_metrics['MAE']:
            best_model = "prophet"
            best_mae = prophet_metrics['MAE']
            
        print(f"Prophet MAE = {prophet_metrics['MAE']}")
        print(f"XGBoost MAE = {xgb_metrics['MAE']}")
        print(f"Best Model = {best_model.upper()}")
        
        # Save model and update router
        key = f"{store_id}_{product_id}"
        router_mapping[key] = {
            'best_model': best_model,
            'metrics': {
                'prophet': prophet_metrics,
                'xgboost': xgb_metrics
            }
        }
        
        aggregated_metrics[key] = router_mapping[key]['metrics'][best_model]
        
        # Save models physically
        if best_model == "xgboost":
            m_xgb.save_model(os.path.join(model_dir, f'xgb_{key}.json'))
        else:
            # save prophet model (you can use json for prophet, or pickle)
            import pickle
            with open(os.path.join(model_dir, f'prophet_{key}.pkl'), 'wb') as f:
                pickle.dump(m_prophet, f)
                
    # Save Router Mapping
    with open(os.path.join(model_dir, 'router.json'), 'w') as f:
        json.dump(router_mapping, f, indent=4)
        
    # Save features list
    with open(os.path.join(model_dir, 'features.json'), 'w') as f:
        json.dump(features, f)
        
    # Pre-calculate demo data for dashboards (last 60 days)
    max_date = df_feat['date'].max()
    demo_df = df_feat[df_feat['date'] >= (max_date - pd.Timedelta(days=60))]
    demo_df.to_csv(os.path.join(model_dir, 'demo_data.csv'), index=False)
    
    df_clean = df[df['date'] >= (df['date'].max() - pd.Timedelta(days=60))]
    df_clean.to_csv(os.path.join(model_dir, 'demo_raw_data.csv'), index=False)
    
    print("\n--- Training Complete ---")
    print(f"Router mapping saved to {os.path.join(model_dir, 'router.json')}")

if __name__ == '__main__':
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', 'sales_data.csv')
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    train_models_and_route(data_path, model_dir)
