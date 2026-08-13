from fastapi import APIRouter, Depends, HTTPException
import pandas as pd
import numpy as np
import os
from pydantic import BaseModel

from backend.core.security import require_role, get_current_user, require_store_access
from backend.models.user import User, RoleEnum

from ml.predict import forecast_demand
from inventory.engine import InventoryDecisionEngine

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ml', 'model')

router = APIRouter(prefix="/inventory", tags=["Inventory & Forecasts"])

class ForecastRequest(BaseModel):
    store_id: str
    product_id: str
    horizon_days: int = 7
    current_stock: int
    safety_stock: int = 20
    lead_time_days: int = 2

@router.post("/recommendations")
def get_inventory_recommendations(
    req: ForecastRequest,
    current_user: User = Depends(require_role([RoleEnum.ADMIN, RoleEnum.MANAGER, RoleEnum.STAFF]))
):
    # Enforce authorization
    require_store_access(req.store_id)(current_user=current_user)
        
    # Generate Forecast
    try:
        forecasts = forecast_demand(
            store_id=req.store_id,
            product_id=req.product_id,
            horizon_days=req.horizon_days,
            model_dir=MODEL_DIR
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    if not forecasts:
        raise HTTPException(status_code=500, detail="Forecast engine failed to return predictions")
        
    forecast_df = pd.DataFrame(forecasts)
    # Run Inventory Engine
    ie = InventoryDecisionEngine()
    total_forecast_demand = forecast_df['predicted_demand'].sum()
    from inventory.schema import ProductInput, InventoryInput, ForecastInput, SupplierInput
    
    prod_in = ProductInput(product_id=req.product_id, service_level=0.95)
    inv_in = InventoryInput(current_stock=req.current_stock, incoming_stock=0, reserved_stock=0)
    for_in = ForecastInput(predicted_demand=int(total_forecast_demand), forecast_horizon=req.horizon_days)
    sup_in = SupplierInput(lead_time_days=req.lead_time_days)
    
    rec = ie.process_single(
        product=prod_in,
        inventory=inv_in,
        forecast=for_in,
        supplier=sup_in,
        raw_forecasts=forecast_df.to_dict(orient='records')
    )
    
    # Custom rule for surplus handling
    surplus_qty = req.current_stock - int(total_forecast_demand)
    if surplus_qty > 0:
        rec.stock_status = "SURPLUS"
        rec.recommended_order = 0
        rec.reason = f"Stock safely exceeds future demand projections. We recommend transferring {surplus_qty} excess units to the Surplus Exchange network."
        
    # Trigger WhatsApp for critical/reorder status
    if rec.stock_status in ["REORDER", "CRITICAL"]:
        from alerts.whatsapp import send_whatsapp_alert
        try:
            # We use background task normally, but for demo we can call it synchronously or securely mock it
            send_whatsapp_alert(
                product_id=req.product_id,
                current_stock=req.current_stock,
                forecast_demand=rec.forecast_demand,
                safety_stock=rec.target_stock - rec.reorder_point if rec.target_stock else 20,
                recommended_reorder=rec.recommended_order,
                force_mock=False
            )
        except Exception as e:
            print(f"Failed to execute whatsapp trigger: {e}")
            
    return {
        "forecasts": forecast_df.to_dict(orient='records'),
        "inventory_recommendation": {
            "stock_status": rec.stock_status,
            "inventory_position": rec.inventory_position,
            "forecast_demand": rec.forecast_demand,
            "target_stock": rec.target_stock,
            "reorder_point": rec.reorder_point,
            "recommended_order": rec.recommended_order,
            "stockout_risk": rec.stockout_risk,
            "overstock_risk": rec.overstock_risk,
            "reason": rec.reason
        },
        "model_metadata": {
            "version": forecasts[0].get('model_version') if forecasts else None,
            "type": forecasts[0].get('selected_model') if forecasts else None
        },
        "request_user": current_user.email
    }
