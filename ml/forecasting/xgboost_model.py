import pandas as pd
import numpy as np
import xgboost as xgb
from .interfaces import ForecastModel

class XGBoostForecaster(ForecastModel):
    def __init__(self, **kwargs):
        self.params = kwargs if kwargs else {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 5,
            'random_state': 42
        }
        self.model = xgb.XGBRegressor(**self.params)
        self.features = []

    def fit(self, train_df: pd.DataFrame, target_col: str, val_df: pd.DataFrame = None):
        drop_cols = ['date', target_col, 'is_outlier', 'safety_stock', 'lead_time_days', 'price', 
                     'store_id', 'product_id', 'category']
        
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        self.features = [c for c in numeric_cols if c not in drop_cols]
        
        X_train = train_df[self.features]
        y_train = train_df[target_col]
        
        if val_df is not None and len(val_df) > 0:
            X_val = val_df[self.features]
            y_val = val_df[target_col]
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            self.model.fit(X_train, y_train, verbose=False)
            
    def predict(self, pred_df: pd.DataFrame):
        X_pred = pred_df[self.features]
        preds = self.model.predict(X_pred)
        return np.maximum(preds, 0)
        
    def save(self, filepath: str):
        # Save features as well so we know what they were
        import json
        config = {'features': self.features, 'params': self.params}
        with open(f"{filepath}_meta.json", 'w') as f:
            json.dump(config, f)
        self.model.save_model(filepath)
        
    def load(self, filepath: str):
        import json
        with open(f"{filepath}_meta.json", 'r') as f:
            config = json.load(f)
            self.features = config['features']
        self.model = xgb.XGBRegressor()
        self.model.load_model(filepath)

    @property
    def model_type(self) -> str:
        return "xgboost"
