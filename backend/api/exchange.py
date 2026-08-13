from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
import uuid
import datetime
import pandas as pd
import math
import os

from backend.core.database import get_db
from backend.core.security import require_role, get_current_user, require_store_access
from backend.models.user import User, RoleEnum
from backend.models.exchange import SurplusListing, ExchangeTransaction, ListingStatus, TransactionStatus

from ml.predict import forecast_demand
from inventory.engine import InventoryDecisionEngine
from inventory.schema import ProductInput, InventoryInput, ForecastInput, SupplierInput
from pydantic import BaseModel

router = APIRouter(prefix="/exchange", tags=["Smart Surplus Exchange"])
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'ml', 'model')

# ================== SCHEMAS ==================
class CreateListingRequest(BaseModel):
    store_id: str
    product_id: str
    available_qty: int
    min_qty: int = 1
    price_per_unit: float
    expiry_days: Optional[int] = None

class ActionTransactionRequest(BaseModel):
    transaction_id: str
    action: str # "ACCEPT", "REJECT", "CONFIRM", "SHIP", "COMPLETE"

class RequestSurplusRequest(BaseModel):
    listing_id: str
    buyer_store_id: str
    requested_qty: int

# ================== MOCK DISTANCE ==================
def get_distance(store_a: str, store_b: str) -> float:
    # A fully functioning system would use a geographic DB (PostGIS) or Coordinates.
    # For now, we mock a deterministic distance based on string lengths.
    if store_a == store_b: return 0.0
    return round(float(abs(len(store_a) - len(store_b)) * 1.5 + 1.2), 1)

# ================== LISTING LOGIC ==================
@router.post("/listings")
def create_surplus_listing(
    req: CreateListingRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([RoleEnum.ADMIN, RoleEnum.MANAGER]))
):
    require_store_access(req.store_id)(current_user=current_user)
    
    # Verify they actually have excess (Surplus Validation via existing engine could go here, 
    # but the prompt specifically noted: "Allow a retailer to create a surplus listing")
    listing_id = f"lst_{uuid.uuid4().hex[:8]}"
    expiry = datetime.datetime.utcnow() + datetime.timedelta(days=req.expiry_days) if req.expiry_days else None
    
    listing = SurplusListing(
        id=listing_id,
        seller_store_id=req.store_id,
        product_id=req.product_id,
        available_qty=req.available_qty,
        min_qty=req.min_qty,
        price_per_unit=req.price_per_unit,
        expiry_date=expiry
    )
    db.add(listing)
    db.commit()
    return {"status": "success", "listing_id": listing_id}

@router.get("/listings")
def get_active_listings(
    store_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([RoleEnum.ADMIN, RoleEnum.MANAGER, RoleEnum.STAFF]))
):
    query = db.query(SurplusListing).filter(SurplusListing.status == ListingStatus.LISTED.value)
    if store_id:
        query = query.filter(SurplusListing.seller_store_id != store_id) # Exclude own listings
        
    listings = query.all()
    out = []
    
    # We dynamically calculate the distance and buyer's match score if store_id provided
    for l in listings:
        dist = get_distance(store_id, l.seller_store_id) if store_id else 0.0
        out.append({
            "listing_id": l.id,
            "seller": l.seller_store_id,
            "product": l.product_id,
            "qty": l.available_qty,
            "price": l.price_per_unit,
            "distance_km": dist,
            "expiry": l.expiry_date.isoformat() if l.expiry_date else None
        })
        
    return {"listings": sorted(out, key=lambda x: x["distance_km"])}

@router.get("/my-listings")
def get_my_listings(
    store_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([RoleEnum.ADMIN, RoleEnum.MANAGER]))
):
    require_store_access(store_id)(current_user=current_user)
    listings = db.query(SurplusListing).filter(SurplusListing.seller_store_id == store_id).all()
    # Fetch transactions on these listings
    listing_ids = [l.id for l in listings]
    txs = db.query(ExchangeTransaction).filter(ExchangeTransaction.listing_id.in_(listing_ids)).all()
    
    return {
        "listings": [{"id": l.id, "product": l.product_id, "qty": l.available_qty, "status": l.status} for l in listings],
        "requests": [{"tx_id": t.id, "listing_id": t.listing_id, "buyer": t.buyer_store_id, "req_qty": t.requested_qty, "status": t.status} for t in txs]
    }

# ================== MATCHING ENGINE ==================
@router.get("/matches/{listing_id}")
def find_matches_for_listing(
    listing_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([RoleEnum.ADMIN, RoleEnum.MANAGER]))
):
    listing = db.query(SurplusListing).filter(SurplusListing.id == listing_id).first()
    if not listing:
        raise HTTPException(404, "Listing not found")
        
    require_store_access(listing.seller_store_id)(current_user=current_user)
    
    # Load all authorized stores interacting with the dataset (Mock fetching from demo_data)
    df = pd.read_csv(os.path.join(MODEL_DIR, "demo_data.csv"))
    all_stores = df["store_id"].unique().tolist()
    valid_stores = [s for s in all_stores if s != listing.seller_store_id]
    
    ie = InventoryDecisionEngine()
    matches = []
    
    for buyer_store in valid_stores:
        try:
            # 1. Run Demand Forecasting for Buyer
            forecasts = forecast_demand(buyer_store, listing.product_id, horizon_days=7, model_dir=MODEL_DIR)
            if not forecasts:
                continue
                
            forecast_df = pd.DataFrame(forecasts)
            total_forecast = int(forecast_df['predicted_demand'].sum())
            
            # Fetch buyer's physical inventory mock - In a real setup, we'd query DB. We assume 5 for calculation.
            mock_buyer_inv = 5
            
            # 2. Run Inventory Intelligence
            prod_in = ProductInput(product_id=listing.product_id)
            inv_in = InventoryInput(current_stock=mock_buyer_inv, incoming_stock=0, reserved_stock=0)
            for_in = ForecastInput(predicted_demand=total_forecast, forecast_horizon=7)
            sup_in = SupplierInput(lead_time_days=2)
            
            rec = ie.process_single(
                product=prod_in, inventory=inv_in, forecast=for_in,
                supplier=sup_in, raw_forecasts=forecast_df.to_dict(orient='records')
            )
            
            # 3. Smart Match Condition: If buyer needs stock
            if rec.stock_status in ["REORDER", "CRITICAL", "LOW_STOCK"]:
                shortage = max(0, rec.target_stock - mock_buyer_inv)
                if shortage > 0:
                    dist = get_distance(listing.seller_store_id, buyer_store)
                    
                    # 4. Scoring Logic Configurable
                    score = 0
                    if rec.stockout_risk == "CRITICAL": score += 50
                    elif rec.stockout_risk == "HIGH": score += 30
                    
                    score -= (dist * 2) # Proximity penalty
                    
                    # Expiry Urgency
                    if listing.expiry_date:
                        days_left = (listing.expiry_date - datetime.datetime.utcnow()).days
                        if days_left <= 3: score += 40 # urgent match!
                        
                    matches.append({
                        "buyer_store": buyer_store,
                        "distance_km": dist,
                        "predicted_shortage": shortage,
                        "stockout_risk": rec.stockout_risk,
                        "recommended_transfer": min(shortage, listing.available_qty),
                        "match_score": round(score, 1)
                    })
        except Exception as e:
            continue
            
    # Rank matches
    matches.sort(key=lambda x: x["match_score"], reverse=True)
    return {"listing_id": listing_id, "matches": matches}

# ================== TRANSACTIONS ==================
@router.post("/requests")
def request_surplus(
    req: RequestSurplusRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([RoleEnum.ADMIN, RoleEnum.MANAGER]))
):
    require_store_access(req.buyer_store_id)(current_user=current_user)
    
    listing = db.query(SurplusListing).filter(SurplusListing.id == req.listing_id).first()
    if not listing or listing.status != ListingStatus.LISTED.value:
        raise HTTPException(400, "Listing unavailable")
        
    if req.requested_qty > listing.available_qty:
        raise HTTPException(400, "Requested quantity exceeds available")
        
    tx_id = f"tx_{uuid.uuid4().hex[:8]}"
    tx = ExchangeTransaction(
        id=tx_id,
        listing_id=req.listing_id,
        buyer_store_id=req.buyer_store_id,
        requested_qty=req.requested_qty
    )
    db.add(tx)
    
    # Instantly lock/remove from surplus exchange explicitly for immediate UI feedback
    listing.status = ListingStatus.MATCHED.value
    
    db.commit()
    return {"status": "success", "transaction_id": tx_id}

@router.post("/transactions/action")
def process_transaction(
    req: ActionTransactionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role([RoleEnum.ADMIN, RoleEnum.MANAGER]))
):
    tx = db.query(ExchangeTransaction).filter(ExchangeTransaction.id == req.transaction_id).first()
    if not tx:
        raise HTTPException(404, "Transaction not found")
        
    listing = tx.listing
    
    # State Machine Logic
    if req.action == "ACCEPT":
        require_store_access(listing.seller_store_id)(current_user=current_user)
        if tx.status != TransactionStatus.REQUESTED.value: raise HTTPException(400, "Invalid state transition")
        tx.status = TransactionStatus.ACCEPTED.value
        listing.status = ListingStatus.MATCHED.value
        
    elif req.action == "CONFIRM":
        # Buyer confirms shipment start
        require_store_access(tx.buyer_store_id)(current_user=current_user)
        if tx.status != TransactionStatus.ACCEPTED.value: raise HTTPException(400, "Invalid state transition")
        tx.status = TransactionStatus.IN_TRANSIT.value
        
    elif req.action == "COMPLETE":
        # Buyer receives goods
        require_store_access(tx.buyer_store_id)(current_user=current_user)
        if tx.status != TransactionStatus.IN_TRANSIT.value: raise HTTPException(400, "Invalid ")
        tx.status = TransactionStatus.COMPLETED.value
        listing.status = ListingStatus.COMPLETED.value
        
        # INVENTORY UPDATE STEP (Mocking it via print log for now, as real physical tables are abstracted)
        print(f"[INVENTORY UPDATE SYSTEM] Seller {listing.seller_store_id} STOCK DECREASE: {tx.requested_qty}")
        print(f"[INVENTORY UPDATE SYSTEM] Buyer {tx.buyer_store_id} STOCK INCREASE: {tx.requested_qty}")
        
    db.commit()
    return {"status": "success", "new_state": tx.status}
