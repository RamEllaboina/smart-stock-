import os
import json
from datetime import datetime
from typing import Dict, Any

class ModelRegistryLite:
    """
    Manages production models, rollback chains, and stores monitoring baselines.
    """
    def __init__(self, registry_dir: str):
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        self.registry_path = os.path.join(registry_dir, 'ops_registry.json')
        self._load()

    def _load(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "active_models": {}, # key: store_product, value: version_id
                "models": {},        # key: version_id, value: metadata
            }

    def _save(self):
        with open(self.registry_path, 'w') as f:
            json.dump(self.data, f, indent=4)
            
    def register_model(self, store_id: str, product_id: str, version_id: str, metadata: dict):
        # Archive previous if present
        key = f"{store_id}_{product_id}"
        prev_version = self.data["active_models"].get(key)
        
        if prev_version and prev_version in self.data["models"]:
            self.data["models"][prev_version]["status"] = "ARCHIVED"
            
        metadata["status"] = "PRODUCTION"
        metadata["version_id"] = version_id
        metadata["store_id"] = store_id
        metadata["product_id"] = product_id
        metadata["registration_date"] = datetime.now().isoformat()
        
        self.data["models"][version_id] = metadata
        self.data["active_models"][key] = version_id
        self._save()
        
    def get_production_model(self, store_id: str, product_id: str) -> dict:
        key = f"{store_id}_{product_id}"
        v_id = self.data["active_models"].get(key)
        if not v_id:
            return None
        return self.data["models"].get(v_id)
        
    def rollback(self, store_id: str, product_id: str) -> bool:
        """
        Rollbacks a product/store to the previous archived model.
        Returns true if successful.
        """
        key = f"{store_id}_{product_id}"
        curr_v_id = self.data["active_models"].get(key)
        if not curr_v_id: return False
        
        # Find latest archived version
        candidates = [v for k, v in self.data["models"].items() if v.get("status") == "ARCHIVED" and v.get("store_id") == store_id and v.get("product_id") == product_id]
        if not candidates:
            return False
            
        # Sort by registration date descending
        candidates.sort(key=lambda x: x.get("registration_date", ""), reverse=True)
        prev_model = candidates[0]
        prev_v_id = prev_model["version_id"]
        
        self.data["models"][curr_v_id]["status"] = "ROLLED_BACK"
        
        self.data["active_models"][key] = prev_v_id
        self.data["models"][prev_v_id]["status"] = "PRODUCTION"
        
        self._save()
        return True
