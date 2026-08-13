import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.forecasting import XGBoostForecaster, ProphetForecaster, BaselineForecaster, ModelRouter
from ml.features import feature_engineering_pipeline, time_series_split

class TestForecastingModels(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2026-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'date': dates,
            'product_id': ['P1'] * 100,
            'store_id': ['S1'] * 100,
            'sales': np.random.randint(10, 50, size=100),
            'price': [10.0] * 100,
            'promotion': [0] * 100
        })
        self.df_feat = feature_engineering_pipeline(self.df)
        self.train_df, self.val_df, self.test_df = time_series_split(self.df_feat, train_prop=0.7, val_prop=0.15)
        
        self.model_dir = os.path.join(os.path.dirname(__file__), 'test_models')
        os.makedirs(self.model_dir, exist_ok=True)
        
    def test_baseline_forecaster(self):
        # Test Naive
        naive = BaselineForecaster(method='naive')
        naive.fit(self.train_df, 'sales')
        preds = naive.predict(self.val_df)
        self.assertEqual(len(preds), len(self.val_df))
        self.assertEqual(preds[0], self.train_df.iloc[-1]['sales'])
        
        # Test Seasonal Naive
        s_naive = BaselineForecaster(method='seasonal_naive')
        s_naive.fit(self.train_df, 'sales')
        preds = s_naive.predict(self.val_df)
        self.assertEqual(len(preds), len(self.val_df))

    def test_xgboost_forecaster(self):
        xgb = XGBoostForecaster(n_estimators=10)
        xgb.fit(self.train_df, 'sales', self.val_df)
        preds = xgb.predict(self.val_df)
        self.assertEqual(len(preds), len(self.val_df))
        self.assertTrue(all(preds >= 0)) # No negatives
        
    def test_prophet_forecaster(self):
        prophet = ProphetForecaster(use_promotions=False)
        prophet.fit(self.train_df, 'sales')
        preds = prophet.predict(self.val_df)
        self.assertEqual(len(preds), len(self.val_df))
        self.assertTrue(all(preds >= 0))
        
    def test_model_router_and_champion(self):
        registry_path = os.path.join(self.model_dir, 'test_registry.json')
        if os.path.exists(registry_path):
            os.remove(registry_path)
            
        router = ModelRouter(registry_path)
        
        # Train and select
        record = router.train_and_select(self.train_df, self.val_df, "S1", "P1", self.model_dir)
        self.assertIsNotNone(record)
        self.assertIn("model_type", record)
        self.assertIn("WAPE", record['metrics'])
        
        selected_type = record['model_type']
        
        # Simulate retraining (Champion vs Challenger tie breaking)
        record2 = router.train_and_select(self.train_df, self.val_df, "S1", "P1", self.model_dir)
        # Should stay champion unless a model miraculously improved > 2%. 
        # Since data is exactly identical, champion should hold its spot!
        self.assertEqual(record2['model_type'], selected_type)
        
        # Check predict_with_fallback
        preds, rec = router.predict_with_fallback("S1_P1", self.val_df)
        self.assertEqual(rec['model_type'], selected_type)
        self.assertEqual(len(preds), len(self.val_df))
        
    def tearDown(self):
        # Cleanup test models
        for f in os.listdir(self.model_dir):
            os.remove(os.path.join(self.model_dir, f))
        os.rmdir(self.model_dir)

if __name__ == '__main__':
    unittest.main()
