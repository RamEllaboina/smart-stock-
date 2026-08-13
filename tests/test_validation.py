import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.validator import DataValidator

class TestDataValidator(unittest.TestCase):
    def setUp(self):
        self.valid_data = pd.DataFrame({
            'date': ['2026-08-01', '2026-08-02', '2026-08-03'],
            'product_id': ['P1', 'P1', 'P1'],
            'store_id': ['S1', 'S1', 'S1'],
            'sales': [10, 15, 20],
            'price': [100.0, 100.0, 100.0]
        })
        
    def test_valid_dataset(self):
        validator = DataValidator(self.valid_data, min_history_length=2)
        report = validator.validate()
        self.assertNotEqual(report.status, 'FAIL')
        self.assertEqual(len(report.errors), 0)
        
    def test_missing_required_columns(self):
        df = self.valid_data.drop(columns=['sales'])
        validator = DataValidator(df)
        report = validator.validate()
        self.assertEqual(report.status, 'FAIL')
        self.assertTrue(any('Missing required columns' in e for e in report.errors))
        
    def test_invalid_data_types(self):
        df = self.valid_data.copy()
        df['sales'] = ['A', 'B', 'C']
        validator = DataValidator(df)
        report = validator.validate()
        self.assertEqual(report.status, 'FAIL')
        self.assertTrue(any('Sales column is not numeric' in e for e in report.errors))
        
    def test_negative_sales(self):
        df = self.valid_data.copy()
        df.loc[0, 'sales'] = -5
        validator = DataValidator(df)
        report = validator.validate()
        self.assertTrue(any('negative sales' in w for w in report.warnings))
        
    def test_future_dates(self):
        df = self.valid_data.copy()
        df.loc[0, 'date'] = '2099-01-01'
        validator = DataValidator(df)
        report = validator.validate()
        self.assertTrue(any('future dates' in w for w in report.warnings))
        
    def test_duplicates(self):
        df = pd.concat([self.valid_data, self.valid_data.iloc[[0]]]).reset_index(drop=True)
        validator = DataValidator(df)
        report = validator.validate()
        self.assertTrue(any('exact duplicate rows' in w for w in report.warnings))
        
if __name__ == '__main__':
    unittest.main()
