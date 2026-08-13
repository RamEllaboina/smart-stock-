import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.anomaly import AnomalyDetector, AnomalyConfig, DataQualityScorer

class TestAnomalyPipeline(unittest.TestCase):
    def setUp(self):
        # 10 days of normal data (around 50)
        self.normal = [50, 52, 48, 51, 49, 53, 50, 48, 51, 50]
        dates = pd.date_range('2026-08-01', periods=10, freq='D')
        
        self.df = pd.DataFrame({
            'date': dates,
            'product_id': 'PROD_1',
            'store_id': 'STORE_1',
            'sales': self.normal,
            'is_promotion': [0]*10,
            'current_stock': [100]*10,
            'price': [10.0]*10
        })

    def test_normal_data_clean(self):
        detector = AnomalyDetector(AnomalyConfig())
        out = detector.process(self.df)
        self.assertFalse(out['is_anomaly'].any())
        
        scorer = DataQualityScorer()
        report = scorer.evaluate(out)
        self.assertEqual(report.total_records, 10)
        self.assertEqual(report.anomalies, 0)
        self.assertEqual(report.data_quality_score, 100.0)
        
    def test_demand_spike_unexplained(self):
        df_spike = self.df.copy()
        df_spike.at[9, 'sales'] = 500 # Unexplained spike
        
        detector = AnomalyDetector(AnomalyConfig())
        out = detector.process(df_spike)
        
        # Row 9 should be anomaly
        self.assertTrue(out.at[9, 'is_anomaly'])
        self.assertEqual(out.at[9, 'anomaly_type'], 'DEMAND_SPIKE')
        
    def test_demand_spike_explained_by_promo(self):
        df_spike = self.df.copy()
        df_spike.at[9, 'sales'] = 400
        df_spike.at[9, 'is_promotion'] = 1 # Explained
        
        detector = AnomalyDetector(AnomalyConfig(promo_threshold_multiplier=5.0))
        out = detector.process(df_spike)
        
        # It shouldn't trigger high severity DEMAND_SPIKE, it might trigger BUSINESS_EVENT or none
        is_high = out.at[9, 'anomaly_severity'] == 'HIGH'
        self.assertFalse(is_high)
        
    def test_negative_inventory(self):
        df_err = self.df.copy()
        df_err.at[5, 'current_stock'] = -50
        
        detector = AnomalyDetector(AnomalyConfig())
        out = detector.process(df_err)
        
        self.assertTrue(out.at[5, 'is_anomaly'])
        self.assertEqual(out.at[5, 'anomaly_type'], 'CRITICAL_DATA_ERROR')
        self.assertEqual(out.at[5, 'anomaly_severity'], 'CRITICAL')
        
        scorer = DataQualityScorer()
        report = scorer.evaluate(out)
        # 1 critical error should reduce quality score
        self.assertTrue(report.data_quality_score < 100)
        self.assertEqual(report.critical_anomalies, 1)

if __name__ == '__main__':
    unittest.main()
