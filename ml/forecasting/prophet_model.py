import pandas as pd
import numpy as np
from prophet import Prophet
from .interfaces import ForecastModel

class ProphetForecaster(ForecastModel):
    def __init__(self, use_promotions: bool = True):
        self.use_promotions = use_promotions
        self.model = None

    def fit(self, train_df: pd.DataFrame, target_col: str, val_df: pd.DataFrame = None):
        history_days = (train_df['date'].max() - train_df['date'].min()).days
        
        yearly_seasonality = history_days > 365
        weekly_seasonality = history_days > 14
        
        self.model = Prophet(
            daily_seasonality=False, # We use daily data, daily seasonality usually means intra-day
            yearly_seasonality=yearly_seasonality, 
            weekly_seasonality=weekly_seasonality
        )
        
        prophet_df = train_df[['date', target_col]].copy()
        prophet_df.columns = ['ds', 'y']
        
        self.has_promotion = False
        if self.use_promotions and 'is_promotion' in train_df.columns:
            if train_df['is_promotion'].nunique() > 1: # Must have variation to use regressor
                prophet_df['is_promotion'] = train_df['is_promotion'].values
                self.model.add_regressor('is_promotion')
                self.has_promotion = True
                
        self.model.fit(prophet_df)
        
    def predict(self, pred_df: pd.DataFrame):
        future = pred_df[['date']].copy()
        future.columns = ['ds']
        
        if self.has_promotion and 'is_promotion' in pred_df.columns:
            future['is_promotion'] = pred_df['is_promotion'].values
        elif self.has_promotion:
            # Fallback if regressor is expected but missing
            future['is_promotion'] = 0
            
        forecast = self.model.predict(future)
        return np.maximum(forecast['yhat'].values, 0)
        
    def save(self, filepath: str):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'has_promotion': self.has_promotion}, f)
            
    def load(self, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.has_promotion = data['has_promotion']

    @property
    def model_type(self) -> str:
        return "prophet"
