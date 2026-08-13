from pydantic import BaseModel
from typing import Optional

class ForecastInput(BaseModel):
    predicted_demand: int
    forecast_horizon: int
    lead_time_demand: Optional[int] = None

class InventoryInput(BaseModel):
    current_stock: int
    reserved_stock: int = 0
    incoming_stock: int = 0

class SupplierInput(BaseModel):
    lead_time_days: int
    minimum_order_quantity: int = 1
    order_multiple: int = 1
    supplier_availability: bool = True

class ProductInput(BaseModel):
    product_id: str
    category: Optional[str] = None
    shelf_life_days: Optional[int] = None

class InventoryConfig(BaseModel):
    service_level: float = 0.95
    review_period_days: int = 7
    default_lead_time_days: int = 3
    enable_moq: bool = True
    enable_order_multiples: bool = True
    enable_shelf_life_constraints: bool = True
    enable_max_inventory: bool = True
    maximum_inventory: Optional[int] = None

class DecisionOutput(BaseModel):
    product_id: str
    current_stock: int
    reserved_stock: int
    incoming_stock: int
    inventory_position: int
    forecast_demand: int
    lead_time_days: int
    lead_time_demand: int
    safety_stock: int
    reorder_point: int
    target_stock: int
    recommended_order: int
    stock_status: str
    stockout_risk: str
    overstock_risk: str
    reason: str
    decision_version: str = "1.0"
