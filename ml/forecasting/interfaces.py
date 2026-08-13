from abc import ABC, abstractmethod
import pandas as pd

class ForecastModel(ABC):
    @abstractmethod
    def fit(self, train_df: pd.DataFrame, target_col: str, val_df: pd.DataFrame = None):
        pass
        
    @abstractmethod
    def predict(self, pred_df: pd.DataFrame):
        pass
        
    @abstractmethod
    def save(self, filepath: str):
        pass
        
    @abstractmethod
    def load(self, filepath: str):
        pass
        
    @property
    @abstractmethod
    def model_type(self) -> str:
        pass
