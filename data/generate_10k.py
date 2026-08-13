import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import datetime

def generate_exact_10000_dataset(output_path):
    np.random.seed(42)
    
    # Configuration
    stores = ['S001', 'S002', 'S003', 'S004', 'S005']
    categories = [
        'Dairy', 'Beverages', 'Bakery', 'Snacks', 'Rice & Grains',
        'Fruits', 'Vegetables', 'Personal Care', 'Household', 'Packaged Foods'
    ]
    
    products_config = [
        # Store, Product, Category, BasePrice, BaseDemand, Volatility, WeeklyEffect(tuple), Trend, Length
        # Long series (750 days)
        ('S001', 'P001', 'Dairy',          50.0,  60, 5,   (1.0,1.0,1.0,1.0,1.0,1.2,1.3), 0.005, 750),
        ('S001', 'P002', 'Dairy',          80.0,  40, 4,   (1.0,1.0,1.0,1.0,1.0,1.1,1.1), -0.002, 750),
        ('S002', 'P003', 'Beverages',      30.0,  80, 10,  (0.9,0.9,0.9,1.0,1.2,1.5,1.5), 0.01, 750),
        ('S002', 'P004', 'Beverages',      90.0,  30, 3,   (1.0,1.0,1.0,1.0,1.0,1.0,1.0), 0.0, 750),
        ('S003', 'P005', 'Bakery',         45.0,  100, 15, (0.8,0.8,0.9,1.0,1.2,1.8,1.6), 0.003, 750),
        ('S003', 'P006', 'Bakery',         100.0, 20, 2,   (0.9,0.9,0.9,0.9,1.0,1.4,1.4), 0.0, 750),
        ('S004', 'P007', 'Snacks',         20.0,  120, 20, (0.7,0.7,0.8,1.0,1.5,1.8,1.5), 0.015, 750),
        ('S004', 'P008', 'Snacks',         60.0,  50, 8,   (0.8,0.8,0.9,1.0,1.3,1.4,1.2), 0.005, 750),
        ('S005', 'P009', 'Rice & Grains',  150.0, 15, 2,   (1.0,1.0,1.0,1.0,1.0,1.1,1.1), -0.001, 750),
        ('S005', 'P010', 'Rice & Grains',  80.0,  25, 3,   (1.0,1.0,1.0,1.0,1.0,1.0,1.0), 0.0, 750),
        
        # Short series (250 days)
        ('S001', 'P011', 'Fruits',         60.0,  45, 6,   (1.0,1.0,1.0,1.0,1.1,1.4,1.4), 0.0, 250),
        ('S001', 'P012', 'Fruits',         120.0, 10, 2,   (1.0,1.0,1.0,1.0,1.0,1.2,1.2), 0.002, 250),
        ('S002', 'P013', 'Vegetables',     30.0,  70, 8,   (1.1,1.1,1.0,1.0,1.0,1.3,1.5), 0.0, 250),
        ('S002', 'P014', 'Vegetables',     80.0,  25, 3,   (1.0,1.0,1.0,1.0,1.0,1.2,1.2), 0.0, 250),
        ('S003', 'P015', 'Personal Care',  150.0, 12, 1,   (1.0,1.0,1.0,1.0,1.0,1.1,1.1), 0.0, 250),
        ('S003', 'P016', 'Personal Care',  250.0,  8, 1,   (1.0,1.0,1.0,1.0,1.0,1.1,1.1), 0.0, 250),
        ('S004', 'P017', 'Household',      300.0,  5, 1,   (1.0,1.0,1.0,1.0,1.0,1.5,1.5), -0.005, 250),
        ('S004', 'P018', 'Household',      80.0,  20, 2,   (1.0,1.0,1.0,1.0,1.0,1.3,1.3), 0.0, 250),
        ('S005', 'P019', 'Packaged Foods', 50.0,  60, 5,   (0.9,0.9,1.0,1.0,1.0,1.2,1.2), 0.0, 250),
        ('S005', 'P020', 'Packaged Foods', 150.0, 18, 2,   (1.0,1.0,1.0,1.0,1.0,1.1,1.1), 0.0, 250),
    ]
    
    end_date = datetime.date(2025, 12, 31)
    
    data = []
    
    for store_id, product_id, category, base_price, base_demand, volatility, weekly_fx, trend, length in products_config:
        start_date = end_date - datetime.timedelta(days=length - 1)
        dates = [start_date + datetime.timedelta(days=i) for i in range(length)]
        
        # Determine store multiplier
        store_mult = 1.0
        if store_id in ['S001', 'S004']: store_mult = 1.3
        elif store_id == 'S003': store_mult = 0.7
        
        for i, d in enumerate(dates):
            dow = d.weekday()
            is_weekend = 1 if dow >= 5 else 0
            
            # Holiday logic (approx 3% chance, plus specific dates)
            # Let's just use some random scattering for holidays across the year + fixed ones like Dec 25
            is_holiday = 1 if (d.month == 12 and d.day == 25) or (d.month == 1 and d.day == 1) or (np.random.random() < 0.02) else 0
            
            # Promotions (approx 5% chance)
            is_promo = 1 if np.random.random() < 0.05 else 0
            
            # Seasonal effect (yearly sine wave, peak in summer for beverages, etc)
            day_of_year = d.timetuple().tm_yday
            season_mult = 1.0
            if category == 'Beverages':
                season_mult = 1 + 0.3 * np.sin(2 * np.pi * (day_of_year - 100) / 365) # Peak in summer
            elif category in ['Fruits', 'Vegetables']:
                season_mult = 1 + 0.15 * np.cos(2 * np.pi * day_of_year / 365) # Peak in winter/spring
                
            # Price (stable but sometimes varies slightly)
            current_price = base_price * (1 if np.random.random() > 0.1 else np.random.choice([0.9, 0.95, 1.05]))
            if is_promo:
                current_price = current_price * 0.8 # 20% discount on promo
                
            price_effect = (base_price / current_price) ** 1.5 # Elasticity
            
            promo_mult = 1.5 if is_promo else 1.0
            holiday_mult = 1.0
            if is_holiday and category in ['Snacks', 'Beverages', 'Bakery', 'Packaged Foods']:
                holiday_mult = 1.8
                
            trend_mult = 1 + (trend * i / 100)
            
            demand_expected = (base_demand * store_mult * weekly_fx[dow] * 
                               season_mult * promo_mult * holiday_mult * 
                               trend_mult * price_effect)
            
            # Random noise
            noise = np.random.normal(0, volatility)
            
            # Occasional spikes (1% chance)
            spike = np.random.uniform(1.8, 2.5) if np.random.random() < 0.01 else 1.0
            
            true_demand = max(0, int((demand_expected + noise) * spike))
            
            # Stock availability
            # Usually stock is high enough (e.g. demand + safety buffer), but sometimes low
            expected_steady_demand = base_demand * store_mult * weekly_fx[dow]
            if np.random.random() < 0.03:
                # Stockout / Low stock scenario
                stock_available = int(expected_steady_demand * np.random.uniform(0.1, 0.7))
            else:
                stock_available = int(expected_steady_demand * np.random.uniform(1.5, 3.0) + 10)
                
            units_sold = min(true_demand, stock_available)
            
            data.append({
                'date': d.strftime('%Y-%m-%d'),
                'store_id': store_id,
                'product_id': product_id,
                'category': category,
                'price': round(current_price, 2),
                'promotion': is_promo,
                'day_of_week': dow,
                'is_weekend': is_weekend,
                'holiday': is_holiday,
                'stock_available': stock_available,
                'units_sold': units_sold
            })
            
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    # Validation stats
    print(f"Total Rows Generated: {len(df)}")
    print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of Stores: {df['store_id'].nunique()}")
    print(f"Number of Products: {df['product_id'].nunique()}")
    print(f"Number of Categories: {df['category'].nunique()}")
    print(f"Average Daily Sales: {df['units_sold'].mean():.2f}")
    print(f"Total Units Sold: {df['units_sold'].sum()}")
    print(f"Promotion %: {(df['promotion'].sum() / len(df)) * 100:.1f}%")
    print(f"Holiday %: {(df['holiday'].sum() / len(df)) * 100:.1f}%")

if __name__ == "__main__":
    generate_exact_10000_dataset('smart_stock_sales_10000.csv')
