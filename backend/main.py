import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uuid
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import settings

# This replaces the entire old main.py and imports our modular routers
from backend.api.auth import router as auth_router
from backend.api.mlops import router as mlops_router
from backend.api.inventory import router as inv_router
from backend.api.exchange import router as exchange_router
 
from backend.core.database import engine, Base
import logging

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Secure Production API for Smart Stock with Multi-Tenancy and RBAC",
    version="2.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/stores", tags=["Static"])
def get_stores():
    import pandas as pd
    try:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml', 'model')
        df = pd.read_csv(os.path.join(model_dir, "demo_data.csv"))
        stores = df["store_id"].unique().tolist()
        return {"stores": [{"id": s} for s in stores]}
    except:
        return {"stores": [{"id": "store_01"}]}

@app.get("/products", tags=["Static"])
def get_products():
    import pandas as pd
    try:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml', 'model')
        df = pd.read_csv(os.path.join(model_dir, "demo_data.csv"))
        products = df["product_id"].unique().tolist()
        return {"products": [{"id": p} for p in products]}
    except:
        return {"products": [{"id": "P001"}]}


# Request ID & Metrics Middleware
@app.middleware("http")
async def add_request_id_and_metrics(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    start_time = time.time()
    
    try:
        response = await call_next(request)
    except Exception as exc:
        logging.error(f"Error {request_id}: {exc}")
        # Standardized Internal error handling happens via Exception handlers, but middleware catches raw
        raise exc
        
    process_time = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    # Security Headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response

# Inclusion of Routers
app.include_router(auth_router)
app.include_router(mlops_router)
app.include_router(inv_router)
app.include_router(exchange_router)

# Health Checks
@app.get("/health/live", tags=["System"])
def liveness_check():
    return {"status": "ok", "service": "Smart Stock API"}

@app.get("/health/ready", tags=["System"])
def readiness_check():
    # Attempt DB connection
    try:
        with engine.connect() as connection:
            pass
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Database connection failed: {e}")

# Pre-populate dummy users if empty
@app.on_event("startup")
def create_initial_users():
    from backend.core.database import SessionLocal
    from backend.models.user import User, RoleEnum
    from backend.core.security import get_password_hash
    db = SessionLocal()
    if db.query(User).count() == 0:
        db.add(User(email="admin@smartstock.local", hashed_password=get_password_hash("admin123"), role=RoleEnum.ADMIN, tenant_id="t1", authorized_stores="*"))
        db.add(User(email="manager@smartstock.local", hashed_password=get_password_hash("manager123"), role=RoleEnum.MANAGER, tenant_id="t1", authorized_stores="store_01,store_02"))
        db.add(User(email="staff@smartstock.local", hashed_password=get_password_hash("staff123"), role=RoleEnum.STAFF, tenant_id="t1", authorized_stores="store_01"))
        db.commit()
    db.close()
