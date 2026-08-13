import pandas as pd
import numpy as np
from .schema import AnomalyConfig

class AnomalyDetector:
    def __init__(self, config: AnomalyConfig = None):
        self.config = config or AnomalyConfig()

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scans dataframe and creates an anomaly augmented dataframe.
        Outputs a copy of DF containing new columns: 
        is_anomaly, anomaly_score, anomaly_type, anomaly_severity, anomaly_reason
        """
        out_df = df.copy()
        
        # Initialize flags
        out_df['is_anomaly'] = False
        out_df['anomaly_score'] = 0.0
        out_df['anomaly_type'] = 'NORMAL'
        out_df['anomaly_severity'] = 'NONE'
        out_df['anomaly_reason'] = ''

        # Store_id parsing
        if 'store_id' not in out_df.columns:
            out_df['store_id'] = 'DEFAULT_STORE'

        # Ensure sorted chronologically per group
        out_df = out_df.sort_values(by=['store_id', 'product_id', 'date']).reset_index(drop=True)
        
        # Calculate context-adjusted bounds group by group
        groups = out_df.groupby(['store_id', 'product_id'])
        
        for name, group in groups:
            store_id, product_id = name
            idx = group.index
            
            # --- Sales Volume & Context Aware ---
            sales = group['sales']
            
            # IQR
            q1 = sales.quantile(0.25)
            q3 = sales.quantile(0.75)
            iqr = q3 - q1
            base_upper = q3 + self.config.iqr_multiplier * iqr
            base_lower = max(0, q1 - self.config.iqr_multiplier * iqr) # No negative sales logic natively
            
            # Context Adjustments
            upper_bound = np.full(len(sales), base_upper)
            if 'is_promotion' in group.columns:
                promo_mask = group['is_promotion'] > 0
                upper_bound[promo_mask] *= self.config.promo_threshold_multiplier
            
            if 'day_of_week' in group.columns:
                weekend_mask = group['day_of_week'] >= 5
                upper_bound[weekend_mask] *= self.config.weekend_threshold_multiplier
                
            # Rolling Z-score
            rolling_mean = sales.rolling(window=self.config.rolling_window, min_periods=1).mean()
            rolling_std = sales.rolling(window=self.config.rolling_window, min_periods=1).std().fillna(1.0)
            rolling_z = np.abs((sales - rolling_mean) / rolling_std)
            
            for i, loc in enumerate(idx):
                val = sales.iloc[i]
                is_promo = group['is_promotion'].iloc[i] if 'is_promotion' in group.columns else 0
                
                # Missing Data Check
                if pd.isna(val):
                    self._flag(out_df, loc, 'MISSING_DATA', 'CRITICAL', 1.0, "Missing sales record.")
                    continue
                
                # Extreme Sales Drops (0 sale where rolling mean is high)
                if val == 0 and rolling_mean.iloc[i] > 10:
                    self._flag(out_df, loc, 'DEMAND_DROP', 'MEDIUM', 0.8, "Sudden 0-demand drop against high rolling average.")
                    continue
                    
                # Anomaly Spikes
                if val > upper_bound[i] or rolling_z.iloc[i] > self.config.rolling_z_threshold:
                    if is_promo:
                        self._flag(out_df, loc, 'BUSINESS_EVENT', 'LOW', 0.4, "High demand aligns with active promotion context.")
                    else:
                        score = min(1.0, (val / (upper_bound[i] + 1e-5))) 
                        severity = 'HIGH' if score > 0.9 else 'MEDIUM'
                        self._flag(out_df, loc, 'DEMAND_SPIKE', severity, score, f"Unexplained demand spike ({val}) exceeds context boundaries.")
                        
                # Inventory Constraint Check
                if self.config.enable_inventory_detection and 'current_stock' in group.columns:
                    stock = group['current_stock'].iloc[i]
                    if stock < 0:
                        self._flag(out_df, loc, 'CRITICAL_DATA_ERROR', 'CRITICAL', 1.0, "Impossible negative inventory value.")
                    elif i > 0:
                        prev_stock = group['current_stock'].iloc[i-1]
                        if stock > prev_stock + 1000: # Heuristic for mass unexplained jump
                            self._flag(out_df, loc, 'INVENTORY_JUMP', 'HIGH', 0.8, "Massive unexplained inventory increase.")
                            
                # Price Error Check
                if self.config.enable_price_detection and 'price' in group.columns:
                    price = group['price'].iloc[i]
                    if price <= 0:
                        self._flag(out_df, loc, 'CRITICAL_DATA_ERROR', 'CRITICAL', 1.0, "Impossible negative or zero price value.")
                    elif i > 0:
                        prev_price = group['price'].iloc[i-1]
                        if prev_price > 0 and (price / prev_price > 2.0 or price / prev_price < 0.2):
                            self._flag(out_df, loc, 'PRICE_ANOMALY', 'HIGH', 0.9, f"Massive price variation ({prev_price} -> {price}).")
                            
        return out_df

    def _flag(self, df, idx, a_type, severity, score, reason):
        # We only override if the new severity is higher or if not yet flagged
        severity_rank = {'NONE': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        current = df.at[idx, 'anomaly_severity']
        
        if severity_rank[severity] > severity_rank[current]:
            df.at[idx, 'is_anomaly'] = True
            df.at[idx, 'anomaly_type'] = a_type
            df.at[idx, 'anomaly_severity'] = severity
            df.at[idx, 'anomaly_score'] = round(score, 2)
            df.at[idx, 'anomaly_reason'] = reason
