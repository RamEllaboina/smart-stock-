import pandas as pd
import numpy as np
from .interfaces import ForecastModel

class BaselineForecaster(ForecastModel):
    def __init__(self, method='seasonal_naive'):
        """
        method: 'naive' or 'seasonal_naive'
        naive: predicts exactly the most recent observation.
        seasonal_naive: predicts the demand from the same weekday in the previous week.
        """
        self.method = method
        self._last_known = None
        self._history = None

    def fit(self, train_df: pd.DataFrame, target_col: str, val_df: pd.DataFrame = None):
        self.target_col = target_col
        if len(train_df) == 0:
            self._last_known = 0
            return
            
        self._last_known = train_df[target_col].iloc[-1]
        
        if self.method == 'seasonal_naive':
            # Store the last 7 days for seasonal naive
            last_7 = train_df.tail(7)
            
            # Map day of week to the last known value for that day
            if 'day_of_week' in last_7.columns:
                self._history = last_7.set_index('day_of_week')[target_col].to_dict()
            else:
                # If no day of week column, just keep the literal last 7 values
                # and cycle them
                self._history = last_7[target_col].values.tolist()
        
    def predict(self, pred_df: pd.DataFrame):
        n = len(pred_df)
        preds = np.zeros(n)
        
        if self.method == 'naive':
            preds = np.full(n, self._last_known if self._last_known is not None else 0)
        elif self.method == 'seasonal_naive':
            for i, row in enumerate(pred_df.to_dict('records')):
                # Try to extract from day_of_week directly
                if 'day_of_week' in row and isinstance(self._history, dict):
                    dow = row['day_of_week']
                    preds[i] = self._history.get(dow, self._last_known)
                elif isinstance(self._history, list) and len(self._history) > 0:
                    # just cycle through the array
                    preds[i] = self._history[i % len(self._history)]
                else:
                    preds[i] = self._last_known if self._last_known is not None else 0
                    
        return np.maximum(preds, 0) # No negative predictions
        
    def save(self, filepath: str):
        # Baseline model usually doesn't need physical saving, 
        # but we can pickle it in a real system.
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'method': self.method, '_last_known': self._last_known, '_history': self._history}, f)
            
    def load(self, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.method = data['method']
            self._last_known = data['_last_known']
            self._history = data['_history']

    @property
    def model_type(self) -> str:
        return f"baseline_{self.method}"
