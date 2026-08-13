import pandas as pd
import numpy as np
import os
from datetime import timedelta, date

def generate_retail_data(file_path):
    np.random.seed(42)
    stores = ['Store_A', 'Store_B']
    products = [
        {'id': 'P_Milk', 'price': 2.50, 'base_demand': 50, 'volatility': 10, 'trend': 0.05, 'category': 'Dairy', 'safety_stock': 20, 'lead_time': 2},
        {'id': 'P_Bread', 'price': 1.50, 'base_demand': 80, 'volatility': 15, 'trend': 0.02, 'category': 'Bakery', 'safety_stock': 30, 'lead_time': 1},
        {'id': 'P_Apples', 'price': 3.00, 'base_demand': 40, 'volatility': 5, 'trend': 0.01, 'category': 'Produce', 'safety_stock': 15, 'lead_time': 3},
        {'id': 'P_Chicken', 'price': 8.00, 'base_demand': 20, 'volatility': 8, 'trend': 0.03, 'category': 'Meat', 'safety_stock': 10, 'lead_time': 2},
        {'id': 'P_Rice', 'price': 12.00, 'base_demand': 15, 'volatility': 5, 'trend': 0.0, 'category': 'Pantry', 'safety_stock': 10, 'lead_time': 5},
    ]
    
    start_date = date(2023, 1, 1)
    # Generate for 1000 days to get exactly 10,000 rows (2 stores * 5 products * 1000 days)
    num_days = 1000
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    data = []
    
    for store in stores:
        # store factor
        store_factor = 1.0 if store == 'Store_A' else 0.7
        for product in products:
            for i, d in enumerate(dates):
                # Day of week effect (higher on weekends)
                dow = d.weekday()
                weekend_boost = 1.3 if dow >= 5 else 1.0
                
                # Seasonality (yearly wave)
                day_of_year = d.timetuple().tm_yday
                seasonality = 1 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
                
                # Promotions (random spikes)
                promo = 1 if np.random.random() < 0.05 else 0
                promo_boost = 1.5 if promo else 1.0
                
                # Base demand calculation
                trend_factor = 1 + (product['trend'] * (i/365))
                
                demand = product['base_demand'] * store_factor * weekend_boost * seasonality * promo_boost * trend_factor
                
                # Add noise
                noise = np.random.normal(0, product['volatility'])
                final_demand = max(0, int(demand + noise))
                
                data.append({
                    'date': d.strftime("%Y-%m-%d"),
                    'store_id': store,
                    'product_id': product['id'],
                    'category': product['category'],
                    'price': product['price'],
                    'sales': final_demand,
                    'promotion': promo,
                    'safety_stock': product['safety_stock'],
                    'lead_time_days': product['lead_time']
                })
                
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Generated data at {file_path}")

if __name__ == '__main__':
    generate_retail_data(r'c:\Users\RAM\OneDrive\Documents\Desktop\demux\smart-stock\data\raw\sales_data.csv')
