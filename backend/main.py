# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from pydantic import BaseModel
from typing import List, Optional
import os
import sys

# Add parent directory to path so we can import 'ml' and 'alerts' when running inside the backend folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.predict import forecast_demand, calculate_inventory_recommendation
from alerts.whatsapp import send_whatsapp_alert

app = FastAPI(title="Smart Stock API", description="Demand Forecasting & Inventory API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml', 'model')

class ForecastRequest(BaseModel):
    store_id: str
    product_id: str
    horizon_days: int = 7
    current_stock: int
    safety_stock: int
    lead_time_days: int

@app.get("/health")
def health_check():
    return {"status": "healthy"}

import pandas as pd

def _get_unique_from_csv(column_name):
    csv_path = os.path.join(MODEL_DIR, "demo_data.csv")
    if not os.path.exists(csv_path):
        return []
    df = pd.read_csv(csv_path, usecols=[column_name])
    return sorted(df[column_name].unique().tolist())

@app.get("/products")
def get_products():
    try:
        product_ids = _get_unique_from_csv("product_id")
        return {"products": [{"id": pid, "name": pid} for pid in product_ids]}
    except Exception as e:
        return {"products": []}

@app.get("/stores")
def get_stores():
    try:
        store_ids = _get_unique_from_csv("store_id")
        return {"stores": [{"id": sid} for sid in store_ids]}
    except Exception as e:
        return {"stores": []}

@app.post("/forecast")
def get_forecast(req: ForecastRequest):
    try:
        forecasts = forecast_demand(req.store_id, req.product_id, req.horizon_days, MODEL_DIR)
        
        if not forecasts:
            raise HTTPException(status_code=404, detail="No historical data found for this product/store")
            
        inventory_rec = calculate_inventory_recommendation(
            req.current_stock, 
            req.safety_stock, 
            req.lead_time_days, 
            forecasts
        )
        
        # Trigger alert if critical
        if inventory_rec['status'] in ["CRITICAL", "REORDER NOW"]:
            send_whatsapp_alert(
                product_id=req.product_id,
                current_stock=req.current_stock,
                forecast_demand=inventory_rec['total_forecast_demand'],
                safety_stock=req.safety_stock,
                recommended_reorder=inventory_rec['recommended_reorder']
            )
        
        return {
            "forecasts": forecasts,
            "inventory_recommendation": inventory_rec
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/performance")
def get_model_performance():
    router_path = os.path.join(MODEL_DIR, "router.json")
    feature_imp_path = os.path.join(MODEL_DIR, "feature_importance.json")
    
    if not os.path.exists(router_path):
        return {"error": "Models not trained yet"}
        
    import json
    with open(router_path, 'r') as f:
        router_mapping = json.load(f)
        
    # Aggregate or just return the mapping
    # To keep simple and compatible, we return router info
    
    return {
        "router_mapping": router_mapping
    }

@app.post("/alerts/test")
def test_alert(product_id: str = "TestProduct"):
    res = send_whatsapp_alert(product_id, 10, 50, 15, 55, force_mock=False)
    return {"status": "Alert triggered", "details": res}
