import os
import json
import traceback
from typing import Dict, Any, List
import pandas as pd
import numpy as np

from ml.evaluate import evaluate_metrics
from .interfaces import ForecastModel
from .xgboost_model import XGBoostForecaster
from .prophet_model import ProphetForecaster
from .baseline_model import BaselineForecaster

class ModelRouter:
    def __init__(self, model_registry_path: str):
        self.registry_path = model_registry_path
        self.registry = self._load_registry()
        self.min_improvement_threshold = 2.0  # Require 2% improvement to switch champion
        self.primary_metric = "WAPE"

    def _load_registry(self) -> Dict:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_registry(self):
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=4)

    def train_and_select(self, p_train: pd.DataFrame, p_val: pd.DataFrame, store_id: str, product_id: str, model_dir: str):
        key = f"{store_id}_{product_id}"
        print(f"--- Processing {key} ---")
        
        target_col = 'sales'
        
        if len(p_train) < 5 or p_train[target_col].sum() == 0:
            print(f"Insufficient historical data for {key}. Defaulting to Baseline.")
            return self._finalize_selection(key, [BaselineForecaster(method='naive')], p_train, p_val, target_col, model_dir)
            
        # Initialize candidates
        candidates: List[ForecastModel] = [
            XGBoostForecaster(),
            ProphetForecaster(),
            BaselineForecaster(method='seasonal_naive'),
            BaselineForecaster(method='naive')
        ]
        
        return self._finalize_selection(key, candidates, p_train, p_val, target_col, model_dir)

    def _finalize_selection(self, key: str, candidates: List[ForecastModel], p_train: pd.DataFrame, p_val: pd.DataFrame, target_col: str, model_dir: str):
        results = []
        models = {}
        
        for candidate in candidates:
            m_type = candidate.model_type
            try:
                candidate.fit(p_train, target_col, p_val)
                preds = candidate.predict(p_val)
                metrics = evaluate_metrics(p_val[target_col], preds)
                
                results.append({
                    'model_type': m_type,
                    'metrics': metrics
                })
                models[m_type] = candidate
            except Exception as e:
                print(f"Training failed for {m_type} on {key}: {e}")
                traceback.print_exc()

        # Selection Logic
        if not results:
            return None
            
        # Sort by Primary Metric (WAPE). If WAPE fails, MAE.
        results.sort(key=lambda x: (x['metrics'].get(self.primary_metric, float('inf')), x['metrics'].get('MAE', float('inf'))))
        
        best_candidate = results[0]
        
        # Tie-braking: if XGBoost and Baseline are very close, maybe we'd prefer baseline? 
        # But we rely on the strict sort and then Champion/Challenger logic.
        
        # Champion vs Challenger Logic
        current_champ = self.registry.get(key)
        final_selected_type = best_candidate['model_type']
        
        if current_champ and current_champ['model_type'] != final_selected_type:
            champ_type = current_champ['model_type']
            # Find champ in current candidates
            current_champ_results = next((r for r in results if r['model_type'] == champ_type), None)
            
            if current_champ_results:
                champ_wape = current_champ_results['metrics'].get(self.primary_metric, float('inf'))
                challenger_wape = best_candidate['metrics'].get(self.primary_metric, float('inf'))
                
                if champ_wape > 0 and ((champ_wape - challenger_wape) / champ_wape) * 100 < self.min_improvement_threshold:
                    # Improvement is less than threshold, keep the champion to avoid thrashing
                    print(f"Model Stability: {final_selected_type} ({challenger_wape:.2f}%) did not beat {champ_type} ({champ_wape:.2f}%) by {self.min_improvement_threshold}%. Retaining Champion.")
                    final_selected_type = champ_type
                    best_candidate = current_champ_results
        
        print("\nModel Comparison Table:")
        print(f"Product/Store: {key}")
        for r in results:
            print(f"{r['model_type'].ljust(20)} | WAPE: {r['metrics']['WAPE']}% | MAE: {r['metrics']['MAE']}")
        print(f"--> Selected Model: {final_selected_type} (Reliability: {best_candidate['metrics']['Reliability']})\n")

        # Save selected model to disk
        selected_model = models[final_selected_type]
        model_filepath = os.path.join(model_dir, f"{final_selected_type}_{key}")
        
        selected_model.save(model_filepath)
        
        version = 1
        if current_champ and current_champ['model_type'] == final_selected_type:
            version = current_champ.get('version', 1) + 1
            
        record = {
            'model_id': f"{key}_{final_selected_type}_v{version}",
            'model_type': final_selected_type,
            'metrics': best_candidate['metrics'],
            'version': version,
            'status': 'production',
            'model_file': model_filepath,
            'all_results': results
        }
        
        self.registry[key] = record
        self.registry[key]['model_id'] = record['model_id']
        self._save_registry()
        
        # MLOps Integration: Sync with ModelRegistryLite and save reference data
        from ml.ops import ModelRegistryLite
        ops_registry = ModelRegistryLite(os.path.join(model_dir, 'registry'))
        
        features = []
        if 'features' in p_train.columns:
            features = list(p_train.columns)
        else:
            features = p_train.select_dtypes(include=[np.number]).columns.tolist()
            
        store_id, product_id = key.split('_', 1)
        
        metadata = {
            'model_type': final_selected_type,
            'version_id': record['model_id'],
            'wape': best_candidate['metrics'].get('WAPE', 0.0),
            'features': features,
            'store_id': store_id,
            'product_id': product_id
        }
        ops_registry.register_model(store_id, product_id, record['model_id'], metadata)
        
        # Save reference data
        if not p_val.empty:
            ref_path = os.path.join(model_dir, f'ref_data_{store_id}_{product_id}.csv')
            p_val.to_csv(ref_path, index=False)
        
        return record

    def predict_with_fallback(self, key: str, pred_df: pd.DataFrame):
        record = self.registry.get(key)
        if not record:
            return None, None
            
        m_type = record['model_type']
        m_filepath = record['model_file']
        
        # Determine model class
        if m_type == 'xgboost':
            model = XGBoostForecaster()
        elif m_type == 'prophet':
            model = ProphetForecaster()
        elif m_type.startswith('baseline_'):
            model = BaselineForecaster(method=m_type.split('_', 1)[1])
        else:
            return None, None
            
        try:
            model.load(m_filepath)
            preds = model.predict(pred_df)
            return preds, record
        except Exception as e:
            print(f"Prediction failed for {m_type} on {key}. Exception: {e}")
            # Fallback logic could jump to baseline statically without history, or Naive
            print(f"Falling back to zero-forecast.")
            return np.zeros(len(pred_df)), None
