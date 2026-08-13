from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import Optional, List
from sqlalchemy.orm import Session
from datetime import datetime

from backend.core.database import get_db
from backend.core.security import require_role, get_current_user, require_store_access
from backend.models.user import User, RoleEnum
import os
import json

from ml.ops import ModelRegistryLite, OpsOrchestrator
from ml.anomaly import AnomalyDetector
from backend.core.config import settings

# This router interacts with the saved artifacts
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ml', 'model')

router = APIRouter(prefix="/monitoring", tags=["Monitoring & MLOps"])

def get_request_id(request: Request):
    return request.state.request_id

@router.get("/health")
def get_ops_health(
    store_id: str, 
    product_id: str, 
    current_user: User = Depends(require_role([RoleEnum.ADMIN, RoleEnum.MANAGER]))
):
    # Enforce multi-tenancy store auth logic via dependency
    require_store_access(store_id)(current_user=current_user)
    
    report_path = os.path.join(MODEL_DIR, "ops_reports", f"report_{store_id}_{product_id}.json")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="MLOps Health Report not found for this product/store")
        
    with open(report_path, 'r') as f:
        return json.load(f)

@router.get("/models")
def get_ops_models(current_user: User = Depends(require_role([RoleEnum.ADMIN]))):
    registry_path = os.path.join(MODEL_DIR, "registry", "ops_registry.json")
    if not os.path.exists(registry_path):
        return {"active_models": {}, "models": {}}
        
    with open(registry_path, 'r') as f:
        return json.load(f)

from pydantic import BaseModel
class RetrainRequest(BaseModel):
    store_id: str
    product_id: str

@router.post("/retrain")
def trigger_retraining(
    req: RetrainRequest, 
    current_user: User = Depends(require_role([RoleEnum.ADMIN]))
):
    report_path = os.path.join(MODEL_DIR, "ops_reports", f"report_{req.store_id}_{req.product_id}.json")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Cannot decide retraining, no ops report exists.")
        
    with open(report_path, 'r') as f:
        report = json.load(f)
        
    decision = report.get("retraining_decision", "NO_ACTION")
    if decision in ["NO_ACTION", "MONITOR"]:
        return {"status": "REJECTED", "reason": f"Decision was {decision}. System is healthy."}
        
    # Simulate saving an audit log
    print(f"AUDIT LOG: ADMIN user {current_user.id} triggered retraining for {req.product_id}")
    
    return {"status": "ACCEPTED", "message": f"Retraining triggered for {req.product_id}.", "request_id": "auto-gen"}

@router.get("/anomalies")
def get_anomalies(
    store_id: Optional[str] = None,
    product_id: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    current_user: User = Depends(require_role([RoleEnum.ADMIN, RoleEnum.MANAGER, RoleEnum.READ_ONLY]))
):
    report_path = os.path.join(MODEL_DIR, 'anomaly_report.json')
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Anomaly report unavailable.")
        
    with open(report_path, 'r') as f:
        data = json.load(f)
        
    results = data.get("results", [])
    
    if store_id:
        require_store_access(store_id)(current_user=current_user)
        results = [r for r in results if r.get('store_id') == store_id]
        
    if product_id:
        results = [r for r in results if r.get('product_id') == product_id]
        
    total = len(results)
    start = (page - 1) * page_size
    end = start + page_size
    
    return {
        "items": results[start:end],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size
    }
