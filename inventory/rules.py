import math

class InventoryRule:
    def evaluate(self, state: dict):
        pass

class InventoryPositionRule(InventoryRule):
    def evaluate(self, state: dict):
        inv = state['inventory']
        state['inventory_position'] = inv.current_stock + inv.incoming_stock - inv.reserved_stock

class LeadTimeDemandRule(InventoryRule):
    def evaluate(self, state: dict):
        forecast_list = state.get('raw_forecasts', [])
        lt = state['supplier'].lead_time_days
        
        # If we have daily forecasts explicitly
        if forecast_list and len(forecast_list) >= lt:
            ltd = sum(f.get('predicted_demand', 0) for f in forecast_list[:lt])
        else:
            # Fallback to average daily extrapolation
            avg_daily = state['forecast'].predicted_demand / state['forecast'].forecast_horizon if state['forecast'].forecast_horizon > 0 else 0
            ltd = int(avg_daily * lt)
            
        state['lead_time_demand'] = ltd

class SafetyStockRule(InventoryRule):
    def evaluate(self, state: dict):
        # Service level approximations (Z-score mapping)
        sl = state['config'].service_level
        if sl >= 0.99: z = 2.33
        elif sl >= 0.95: z = 1.645
        else: z = 1.28
        
        # Approximate demand variability using Poisson assumption if historical sigma unprovided
        ltd = state['lead_time_demand']
        sigma = math.sqrt(ltd) if ltd > 0 else 1.0
        
        state['safety_stock'] = int(math.ceil(z * sigma))

class ReorderPointRule(InventoryRule):
    def evaluate(self, state: dict):
        state['reorder_point'] = state['lead_time_demand'] + state['safety_stock']

class ReorderDecisionRule(InventoryRule):
    def evaluate(self, state: dict):
        pos = state['inventory_position']
        rop = state['reorder_point']
        state['should_reorder'] = pos <= rop

class TargetStockRule(InventoryRule):
    def evaluate(self, state: dict):
        fs = state['forecast']
        avg_daily = fs.predicted_demand / fs.forecast_horizon if fs.forecast_horizon > 0 else 0
        review_demand = avg_daily * state['config'].review_period_days
        
        target = int(math.ceil(review_demand + state['lead_time_demand'] + state['safety_stock']))
        state['target_stock'] = target

class OrderQuantityRule(InventoryRule):
    def evaluate(self, state: dict):
        if not state['should_reorder'] or not state['supplier'].supplier_availability:
            state['recommended_order'] = 0
            if not state['should_reorder']:
                state['reasons'].append("Inventory position is above reorder point.")
            else:
                state['reasons'].append("Supplier is unavailable.")
            return
            
        target = state['target_stock']
        pos = state['inventory_position']
        
        order_qty = max(0, target - pos)
        
        config = state['config']
        supplier = state['supplier']
        
        # Max inventory constraint
        if config.enable_max_inventory and config.maximum_inventory:
            max_order_allowed = max(0, config.maximum_inventory - pos)
            if order_qty > max_order_allowed:
                order_qty = max_order_allowed
                state['reasons'].append(f"Order capped by max inventory limit ({config.maximum_inventory}).")
                
        # MOQ rule
        if config.enable_moq and supplier.minimum_order_quantity > 1:
            if 0 < order_qty < supplier.minimum_order_quantity:
                order_qty = supplier.minimum_order_quantity
                state['reasons'].append(f"Order increased to meet minimum order quantity ({supplier.minimum_order_quantity}).")
                
        # Order multiples rule
        if config.enable_order_multiples and supplier.order_multiple > 1:
            if order_qty > 0:
                remainder = order_qty % supplier.order_multiple
                if remainder != 0:
                    order_qty += (supplier.order_multiple - remainder)
                    state['reasons'].append(f"Order rounded up to match supplier multiple ({supplier.order_multiple}).")
                    
        # Shelf life rule
        product = state['product']
        if config.enable_shelf_life_constraints and product.shelf_life_days:
            avg_daily = state['forecast'].predicted_demand / state['forecast'].forecast_horizon if state['forecast'].forecast_horizon > 0 else 0
            max_consumption = int(avg_daily * product.shelf_life_days)
            # Cannot safely order more than we can sell before expiry
            if order_qty > max_consumption:
                order_qty = max_consumption
                if order_qty == 0:
                    state['reasons'].append("Order set to zero because shelf life will expire before demand materializes.")
                else:    
                    state['reasons'].append(f"Order strictly capped by perishable shelf-life limit ({max_consumption} units).")
                
        if order_qty > 0:
            # Prevent pushing general reason if we had to adjust to 0 due to max limitations
            state['reasons'].append(f"Inventory position ({pos}) dropped below reorder point ({state['reorder_point']}).")
            
        state['recommended_order'] = order_qty

class StatusClassificationRule(InventoryRule):
    def evaluate(self, state: dict):
        pos = state['inventory_position']
        rop = state['reorder_point']
        ss = state['safety_stock']
        config = state['config']
        
        if config.enable_max_inventory and config.maximum_inventory and pos > config.maximum_inventory:
            state['stock_status'] = "OVERSTOCK"
        elif pos <= ss:
            state['stock_status'] = "CRITICAL"
        elif pos <= rop:
            state['stock_status'] = "REORDER"
        elif pos <= (rop + (state['target_stock'] - rop) * 0.3):
            state['stock_status'] = "LOW_STOCK"
        else:
            state['stock_status'] = "HEALTHY"

class RiskEvaluationRule(InventoryRule):
    def evaluate(self, state: dict):
        pos = state['inventory_position']
        demand = state['forecast'].predicted_demand
        
        # Stockout Risk
        if pos <= demand * 0.2:
            state['stockout_risk'] = "CRITICAL"
        elif pos <= demand * 0.5:
            state['stockout_risk'] = "HIGH"
        elif pos <= demand * 1.0:
            state['stockout_risk'] = "MEDIUM"
        else:
            state['stockout_risk'] = "LOW"
            
        # Overstock Risk
        if pos >= demand * 3:
            state['overstock_risk'] = "HIGH"
        elif pos >= demand * 2:
            state['overstock_risk'] = "MEDIUM"
        else:
            state['overstock_risk'] = "LOW"
