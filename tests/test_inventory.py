import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inventory import (
    InventoryDecisionEngine, InventoryConfig,
    ProductInput, InventoryInput, SupplierInput, ForecastInput
)

class TestInventoryEngine(unittest.TestCase):
    def setUp(self):
        self.engine = InventoryDecisionEngine()
        
    def test_healthy_inventory(self):
        decision = self.engine.process_single(
            product=ProductInput(product_id="Prod1"),
            inventory=InventoryInput(current_stock=100, incoming_stock=20), # pos = 120
            supplier=SupplierInput(lead_time_days=2),
            forecast=ForecastInput(forecast_horizon=7, predicted_demand=70) # 10/day
        )
        # lead_time_demand = 10 * 2 = 20
        # safety_stock ~ something low
        # reorder_point ~ 25
        # Target stock ~ demands + rop
        
        self.assertEqual(decision.inventory_position, 120)
        self.assertEqual(decision.lead_time_demand, 20)
        self.assertEqual(decision.stock_status, "HEALTHY")
        self.assertEqual(decision.recommended_order, 0)
        
    def test_reorder_triggered(self):
        decision = self.engine.process_single(
            product=ProductInput(product_id="Prod2"),
            inventory=InventoryInput(current_stock=10, incoming_stock=0),
            supplier=SupplierInput(lead_time_days=3),
            forecast=ForecastInput(forecast_horizon=7, predicted_demand=70) # 10/day
        )
        # lead time demand = 30
        # pos = 10 -> well below rop
        self.assertTrue(decision.inventory_position < decision.reorder_point)
        self.assertIn(decision.stock_status, ["CRITICAL", "REORDER", "LOW_STOCK"])
        self.assertTrue(decision.recommended_order > 0)
        
    def test_max_inventory_constraint(self):
        engine = InventoryDecisionEngine(InventoryConfig(maximum_inventory=50))
        decision = engine.process_single(
            product=ProductInput(product_id="Prod3"),
            inventory=InventoryInput(current_stock=10, incoming_stock=0),
            supplier=SupplierInput(lead_time_days=2),
            forecast=ForecastInput(forecast_horizon=7, predicted_demand=210) # 30/day
        )
        
        # Target stock would naturally be high (30*7 = 210 + etc)
        # But max inventory is 50.
        # Order should be capped at max_inv - pos = 50 - 10 = 40
        self.assertTrue(decision.recommended_order <= 40)
        self.assertIn("capped by max inventory", decision.reason)
        
    def test_moq_constraint(self):
        decision = self.engine.process_single(
            product=ProductInput(product_id="Prod4"),
            inventory=InventoryInput(current_stock=5),
            supplier=SupplierInput(lead_time_days=2, minimum_order_quantity=100),
            forecast=ForecastInput(forecast_horizon=7, predicted_demand=35) # 5/day
        )
        # Target is low, maybe we only need 20 units.
        # But MOQ is 100
        self.assertTrue(decision.recommended_order >= 100)
        self.assertIn("minimum order quantity", decision.reason)
        
    def test_order_multiple_constraint(self):
        decision = self.engine.process_single(
            product=ProductInput(product_id="Prod5"),
            inventory=InventoryInput(current_stock=5),
            supplier=SupplierInput(lead_time_days=2, minimum_order_quantity=10, order_multiple=12),
            forecast=ForecastInput(forecast_horizon=7, predicted_demand=35) # 5/day
        )
        # Target stock - pos = order_qty. 
        # If it needed 22, it will round to 24 due to order multiple of 12.
        self.assertEqual(decision.recommended_order % 12, 0)
        
    def test_shelf_life_constraint(self):
        decision = self.engine.process_single(
            product=ProductInput(product_id="Milk", shelf_life_days=3),
            inventory=InventoryInput(current_stock=2),
            supplier=SupplierInput(lead_time_days=1, minimum_order_quantity=1000), # highly unrealistic MOQ
            forecast=ForecastInput(forecast_horizon=7, predicted_demand=70) # 10/day
        )
        # Shelf life 3 days -> max consumption is 3 * 10 = 30
        # Even with MOQ of 1000, shelf life rule caps it strictly to prevent massive waste
        self.assertTrue(decision.recommended_order <= 30)
        self.assertIn("capped by perishable shelf-life", decision.reason)
        
    def test_stockout_risk(self):
        decision = self.engine.process_single(
            product=ProductInput(product_id="Prod6"),
            inventory=InventoryInput(current_stock=5), # demand is 100, we have 5
            supplier=SupplierInput(lead_time_days=5),
            forecast=ForecastInput(forecast_horizon=7, predicted_demand=100)
        )
        self.assertEqual(decision.stockout_risk, "CRITICAL")
        self.assertEqual(decision.stock_status, "CRITICAL")
        
    def test_zero_demand(self):
        decision = self.engine.process_single(
            product=ProductInput(product_id="Prod7"),
            inventory=InventoryInput(current_stock=20),
            supplier=SupplierInput(lead_time_days=5),
            forecast=ForecastInput(forecast_horizon=7, predicted_demand=0)
        )
        self.assertEqual(decision.recommended_order, 0)
        self.assertEqual(decision.stockout_risk, "LOW")

if __name__ == '__main__':
    unittest.main()
