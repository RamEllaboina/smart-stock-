import pandas as pd
import numpy as np
from typing import List, Dict, Any
from .schema import REQUIRED_COLUMNS, OPTIONAL_COLUMNS, ECOMMERCE_MAPPING
from .report import ValidationReport

class DataValidator:
    def __init__(self, df: pd.DataFrame, min_history_length: int = 30):
        self.df = df.copy()
        self.min_history_length = min_history_length
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.infos: List[str] = []
        self.statistics: Dict[str, Any] = {}
        
        # We rename columns initially to validate schema correctly
        self.df = self.df.rename(columns=ECOMMERCE_MAPPING)
        
    def _add_error(self, msg: str):
        self.errors.append(msg)
        
    def _add_warning(self, msg: str):
        self.warnings.append(msg)
        
    def _add_info(self, msg: str):
        self.infos.append(msg)

    def validate(self) -> ValidationReport:
        self.statistics['rows'] = len(self.df)
        
        if not self.check_schema():
            # If schema fails, return early
            return self._build_report()
            
        self.check_types()
        self.check_missing()
        self.check_duplicates()
        if not self.errors:
            self.check_dates()
            self.check_sales()
            self.check_outliers()
            self.check_continuity_and_history()
            self.check_leakage()
        
        self.statistics['products'] = self.df['product_id'].nunique() if 'product_id' in self.df.columns else 0
        self.statistics['stores'] = self.df['store_id'].nunique() if 'store_id' in self.df.columns else 0
        
        if 'date' in self.df.columns and not self.df['date'].isna().all():
            min_date = self.df['date'].min()
            max_date = self.df['date'].max()
            if isinstance(min_date, pd.Timestamp):
                self.statistics['date_range'] = (min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'))
            else:
                self.statistics['date_range'] = (str(min_date), str(max_date))
                
        return self._build_report()
        
    def _build_report(self) -> ValidationReport:
        status = "PASS"
        if self.errors:
            status = "FAIL"
        elif self.warnings:
            status = "WARNING"
            
        return ValidationReport(status, self.errors, self.warnings, self.infos, self.statistics, self.df)

    def check_schema(self) -> bool:
        missing_required = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_required:
            self._add_error(f"Missing required columns: {', '.join(missing_required)}")
            return False
            
        missing_optional = [col for col in OPTIONAL_COLUMNS if col not in self.df.columns]
        if missing_optional:
            self._add_info(f"Missing optional columns: {', '.join(missing_optional)}")
            
        # Check if only one store exists
        if 'store_id' in self.df.columns and self.df['store_id'].nunique() == 1:
            self._add_info("Dataset contains only one store")
            
        if 'promotion' not in self.df.columns and 'Discount_Applied' not in self.df.columns:
            self._add_info("No promotions column available")
            
        return True

    def check_types(self):
        # Check sales
        if not pd.api.types.is_numeric_dtype(self.df['sales']):
            self._add_error("Sales column is not numeric.")
            
        # Check price
        if 'price' in self.df.columns and not pd.api.types.is_numeric_dtype(self.df['price']):
            self._add_error("Price column is not numeric.")
            
        # Check dates
        try:
            self.df['date'] = pd.to_datetime(self.df['date'], errors='raise')
            # Format is valid
        except Exception:
            self._add_warning("Date column contains unparseable values.")
            self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')

    def check_missing(self):
        missing = self.df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                pct = (count / len(self.df)) * 100
                msg = f"Column '{col}' has {count} missing values ({pct:.2f}%)."
                # Report as warning - preprocessing handles it.
                self._add_warning(msg)

    def check_duplicates(self):
        exact_dupes = self.df.duplicated().sum()
        if exact_dupes > 0:
            self._add_warning(f"Found {exact_dupes} exact duplicate rows.")
            
        check_cols = ['date', 'product_id']
        if 'store_id' in self.df.columns:
            check_cols.append('store_id')
            
        subset_dupes = self.df.duplicated(subset=check_cols).sum()
        if subset_dupes > 0:
            self._add_warning(f"Found {subset_dupes} duplicate rows based on {check_cols}.")

    def check_dates(self):
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            return
            
        # Future dates
        future_dates = (self.df['date'] > pd.Timestamp.now()).sum()
        if future_dates > 0:
            self._add_warning(f"Found {future_dates} rows with future dates.")
        
    def check_sales(self):
        if not pd.api.types.is_numeric_dtype(self.df['sales']):
            return
            
        negatives = (self.df['sales'] < 0).sum()
        if negatives > 0:
            self._add_warning(f"Found {negatives} rows with negative sales. May represent returns.")
            
        zeros = (self.df['sales'] == 0).sum()
        if zeros > 0:
            self._add_info(f"Found {zeros} rows with zero sales.")
            
        if np.isinf(self.df['sales']).sum() > 0:
            self._add_warning("Found infinite values in sales.")
            
    def check_outliers(self):
        if not pd.api.types.is_numeric_dtype(self.df['sales']):
            return
            
        group_cols = ['product_id']
        if 'store_id' in self.df.columns:
            group_cols.append('store_id')
            
        # IQR method for spikes
        Q1 = self.df.groupby(group_cols)['sales'].transform(lambda x: x.quantile(0.25) if len(x.dropna()) > 3 else np.nan)
        Q3 = self.df.groupby(group_cols)['sales'].transform(lambda x: x.quantile(0.75) if len(x.dropna()) > 3 else np.nan)
        IQR = Q3 - Q1
        upper = Q3 + 1.5 * IQR
        
        outliers = (self.df['sales'] > upper).sum()
        if outliers > 0:
            self._add_warning(f"Found {outliers} potential demand outliers (spikes) using IQR method.")

    def check_continuity_and_history(self):
        if not pd.api.types.is_datetime64_any_dtype(self.df['date']):
            return
            
        group_cols = ['product_id']
        if 'store_id' in self.df.columns:
            group_cols.append('store_id')
            
        grouped = self.df.groupby(group_cols)
        
        for name, group in grouped:
            if isinstance(name, tuple):
                name_str = "-".join(map(str, name))
            else:
                name_str = str(name)
                
            history_len = len(group)
            if history_len < self.min_history_length:
                self._add_warning(f"Insufficient history for {name_str}: {history_len} records (Min: {self.min_history_length}). Status: INSUFFICIENT_DATA")
                
            if history_len > 1:
                group = group.sort_values('date')
                min_d = group['date'].min()
                max_d = group['date'].max()
                expected = (max_d - min_d).days + 1
                actual = group['date'].nunique()
                missing = expected - actual
                if missing > 0:
                    pct = (missing / expected) * 100
                    if pct > 10:
                        self._add_warning(f"Item {name_str} has {missing} missing dates ({pct:.2f}% missing rate).")
                        
    def check_leakage(self):
        suspicious_cols = [c for c in self.df.columns if ('future' in c.lower() or 'target' in c.lower() or 'test' in c.lower()) and c != 'sales' and c != 'category']
        if suspicious_cols:
            self._add_error(f"Potential data leakage detected in raw data. Suspicious columns: {suspicious_cols}")
