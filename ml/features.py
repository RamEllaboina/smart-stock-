import pandas as pd
import numpy as np

def create_calendar_features(df):
    df = df.copy()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_month'] = df['date'].dt.day
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    return df

def create_lag_features(df, lags=[1, 7, 14, 28]):
    df = df.copy()
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby(['store_id', 'product_id'])['sales'].shift(lag)
    return df

def create_rolling_features(df, windows=[7, 14, 28]):
    df = df.copy()
    for w in windows:
        df[f'rolling_mean_{w}'] = df.groupby(['store_id', 'product_id'])['sales'].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        if w in [7, 28]:
            df[f'rolling_std_{w}'] = df.groupby(['store_id', 'product_id'])['sales'].transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).std())
    return df

def feature_engineering_pipeline(df):
    df = create_calendar_features(df)
    df = create_lag_features(df)
    df = create_rolling_features(df)
    
    # Drop rows with NaN values created by lag and rolling window operations
    df = df.dropna().reset_index(drop=True)
    
    # Categorical encoding (store_id, product_id, category)
    # For XGBoost, we can use label encoding or just get_dummies since stores/products are small
    encoded_cols = pd.get_dummies(df[['store_id', 'product_id', 'category']], drop_first=False)
    df = pd.concat([df, encoded_cols], axis=1)
    
    return df

def prepare_features_for_prediction(historical_df, forecast_date, store_id, product_id):
    """
    Given historical DataFrame (with all features), generate the exact feature row for forecast_date.
    This effectively requires shifting the most recent known values to form lag and rolling features.
    """
    # For a naive step-by-step prediction over a horizon, we need to iteratively predict and append.
    pass
