import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_idx = y_true != 0
    if not np.any(non_zero_idx):
        return 0.0
    return np.mean(np.abs((y_true[non_zero_idx] - y_pred[non_zero_idx]) / y_true[non_zero_idx])) * 100

def weighted_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    sum_true = np.sum(y_true)
    if sum_true == 0:
        sum_true = 1e-6
    return (np.sum(np.abs(y_true - y_pred)) / sum_true) * 100

def get_reliability_score(wape):
    if wape < 10:
        return 'Excellent'
    elif wape < 20:
        return 'Good'
    elif wape < 35:
        return 'Moderate'
    return 'Low'

def evaluate_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {'MAE': 0, 'RMSE': 0, 'MAPE': 0, 'WAPE': 0, 'Reliability': 'Low'}
        
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    wape = weighted_absolute_percentage_error(y_true, y_pred)
    
    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2),
        'WAPE': round(wape, 2),
        'Reliability': get_reliability_score(wape)
    }
