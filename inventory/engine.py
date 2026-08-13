from typing import List, Dict
import logging
from .schema import DecisionOutput, ForecastInput, InventoryInput, SupplierInput, ProductInput, InventoryConfig
from .rules import (
    InventoryPositionRule, LeadTimeDemandRule, SafetyStockRule, 
    ReorderPointRule, ReorderDecisionRule, TargetStockRule, 
    OrderQuantityRule, StatusClassificationRule, RiskEvaluationRule
)

logger = logging.getLogger(__name__)

class InventoryDecisionEngine:
    def __init__(self, config: InventoryConfig = None):
        self.config = config or InventoryConfig()
        self.rules = [
            InventoryPositionRule(),
            LeadTimeDemandRule(),
            SafetyStockRule(),
            ReorderPointRule(),
            ReorderDecisionRule(),
            TargetStockRule(),
            OrderQuantityRule(),
            StatusClassificationRule(),
            RiskEvaluationRule()
        ]
        
    def process_single(
        self, 
        product: ProductInput, 
        inventory: InventoryInput, 
        forecast: ForecastInput, 
        supplier: SupplierInput,
        raw_forecasts: List[Dict] = None
    ) -> DecisionOutput:
        
        state = {
            'product': product,
            'inventory': inventory,
            'forecast': forecast,
            'supplier': supplier,
            'config': self.config,
            'raw_forecasts': raw_forecasts or [],
            'reasons': []
        }
        
        for rule in self.rules:
            rule.evaluate(state)
            
        reason_str = " | ".join(state['reasons']) if state['reasons'] else "Stock levels are adequate."
        
        # Logging integration
        logger.info(f"Product {product.product_id}")
        logger.info(f"Inventory Position = {state['inventory_position']}")
        logger.info(f"Reorder Point = {state['reorder_point']}")
        logger.info(f"Decision = {'REORDER' if state['should_reorder'] else 'NO_REORDER'}")
        logger.info(f"Recommended Order = {state['recommended_order']}")
        logger.info(f"Reason = {reason_str}")
            
        return DecisionOutput(
            product_id=product.product_id,
            current_stock=inventory.current_stock,
            reserved_stock=inventory.reserved_stock,
            incoming_stock=inventory.incoming_stock,
            inventory_position=state['inventory_position'],
            forecast_demand=forecast.predicted_demand,
            lead_time_days=supplier.lead_time_days,
            lead_time_demand=state['lead_time_demand'],
            safety_stock=state['safety_stock'],
            reorder_point=state['reorder_point'],
            target_stock=state['target_stock'],
            recommended_order=state['recommended_order'],
            stock_status=state['stock_status'],
            stockout_risk=state['stockout_risk'],
            overstock_risk=state['overstock_risk'],
            reason=reason_str,
            decision_version="1.0"
        )
        
    def process_batch(self, items: List[Dict]) -> List[DecisionOutput]:
        """
        Process multiple products efficiently.
        Expects a list of dictionaries, each containing instantiated ProductInput, 
        InventoryInput, ForecastInput, SupplierInput.
        """
        results = []
        for item in items:
            results.append(
                self.process_single(
                    product=item['product'],
                    inventory=item['inventory'],
                    forecast=item['forecast'],
                    supplier=item['supplier'],
                    raw_forecasts=item.get('raw_forecasts')
                )
            )
        return results
