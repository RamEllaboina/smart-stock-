import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features import (
    create_lag_features,
    create_rolling_features,
    create_calendar_features,
    create_trend_features,
    create_promotion_features,
    create_price_features,
    time_series_split,
    feature_engineering_pipeline,
    FEATURE_CONFIG
)

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2026-01-01', periods=30, freq='D')
        self.df = pd.DataFrame({
            'date': dates,
            'product_id': ['P1'] * 30,
            'store_id': ['S1'] * 30,
            'sales': np.arange(1, 31, dtype=float), # 1 to 30
            'price': [10.0] * 30,
            'promotion': [0] * 15 + [1] * 15
        })
        
        # Add another product to test isolation
        df2 = self.df.copy()
        df2['product_id'] = 'P2'
        df2['sales'] = np.arange(100, 130, dtype=float)
        self.mixed_df = pd.concat([self.df, df2]).reset_index(drop=True)

    def test_lag_features(self):
        df_lags = create_lag_features(self.mixed_df)
        
        # Test lag_1 for P1
        p1_data = df_lags[df_lags['product_id'] == 'P1'].reset_index(drop=True)
        self.assertTrue(np.isnan(p1_data.loc[0, 'lag_1']))
        self.assertEqual(p1_data.loc[0, 'has_lag_1'], 0)
        self.assertEqual(p1_data.loc[1, 'lag_1'], 1.0)
        self.assertEqual(p1_data.loc[1, 'has_lag_1'], 1)
        
        # Test lag isolation (lag 1 of first row of P2 should be NaN, not last row of P1)
        p2_data = df_lags[df_lags['product_id'] == 'P2'].reset_index(drop=True)
        self.assertTrue(np.isnan(p2_data.loc[0, 'lag_1']))
        self.assertEqual(p2_data.loc[1, 'lag_1'], 100.0)
        
        # Zero sales vs Missing History
        df_zero = self.df.copy()
        df_zero.loc[0, 'sales'] = 0.0
        df_lags_zero = create_lag_features(df_zero)
        self.assertEqual(df_lags_zero.loc[1, 'lag_1'], 0.0)
        self.assertEqual(df_lags_zero.loc[1, 'has_lag_1'], 1) # history exists, it's just 0

    def test_rolling_feature_leakage(self):
        # Specific test proving rolling features don't include current day sales
        df_roll = create_rolling_features(self.df)
        
        # On day 3 (idx 2), sales are [1, 2, 3]. mean(window=3) including today would be 2.
        # But shifted rolling mean(3) on day 3 should be mean([1, 2]) = 1.5
        # shifted (sales) on day 3 is 2. The mean of shifted values up to day 3 is mean(2, 1) = 1.5.
        
        # Let's see: original sales: [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Day 2 (idx 1): sales=2, lag_1=1. rolling_mean_7 = mean(1) = 1.0
        # Day 3 (idx 2): sales=3, lags=[1,2]. rolling_mean_7 = mean(1,2) = 1.5
        # Day 8 (idx 7): sales=8, lags=[1,2,3,4,5,6,7]. mean = 4.0
        
        self.assertEqual(df_roll.loc[1, 'rolling_mean_7'], 1.0)
        self.assertEqual(df_roll.loc[2, 'rolling_mean_7'], 1.5)
        self.assertEqual(df_roll.loc[7, 'rolling_mean_7'], 4.0)
        
        # The target current sales value should NOT be in the rolling mean.
        # Current sales for idx 7 is 8.0. If it was included, mean would be > 4.
        self.assertNotIn(self.df.loc[7, 'sales'], df_roll.loc[7, ['rolling_mean_7']])

    def test_price_and_trend(self):
        # We need rolling features first for trend
        df_roll = create_rolling_features(self.df)
        df_trend = create_trend_features(df_roll)
        self.assertIn('trend_ratio', df_trend.columns)
        
        df_price = create_price_features(self.df)
        self.assertIn('price_change', df_price.columns)
        
    def test_optional_columns(self):
        # Dropping promotion and price shouldn't break the pipeline
        df_no_opt = self.df.drop(columns=['promotion', 'price'])
        
        df_promo = create_promotion_features(df_no_opt)
        self.assertNotIn('is_promotion', df_promo.columns)
        
        df_pipe = feature_engineering_pipeline(df_no_opt)
        self.assertTrue('lag_1' in df_pipe.columns)
        self.assertTrue('rolling_mean_7' in df_pipe.columns)

    def test_chronological_split(self):
        train, val, test = time_series_split(self.df, 0.7, 0.15)
        self.assertEqual(len(train), 21)
        self.assertEqual(len(val), 4) # 30 * 0.85 = 25.5
        self.assertEqual(len(test), 5)
        
        # Ensure strict chronology
        train_max = train['date'].max()
        val_min = val['date'].min()
        val_max = val['date'].max()
        test_min = test['date'].min()
        
        self.assertTrue(train_max < val_min)
        self.assertTrue(val_max < test_min)

if __name__ == '__main__':
    unittest.main()
