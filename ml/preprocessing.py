import pandas as pd
import numpy as np

def clean_data(df):
    """"
    Clean and validate the raw sales data.
    """
    df = df.copy()
    
    # Auto-map columns if using the Global E-Commerce dataset
    ecommerce_mapping = {
        'Date': 'date',
        'Location': 'store_id',
        'Product_Category': 'category',
        'Product_Name': 'product_id',
        'Unit_Price': 'price',
        'Quantity': 'sales'
    }
    df = df.rename(columns=ecommerce_mapping)
    
    # 1. Date conversion
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # 2. Invalid-date detection
    df = df.dropna(subset=['date'])
    
    # 3. Aggregate Daily Sales
    agg_funcs = {'sales': 'sum', 'price': 'mean'}
    if 'Discount_Applied' in df.columns:
        agg_funcs['Discount_Applied'] = 'max'
        
    group_cols = ['date', 'store_id', 'product_id']
    if 'category' in df.columns:
        group_cols.append('category')
        
    df = df.groupby(group_cols).agg({k: v for k, v in agg_funcs.items() if k in df.columns}).reset_index()
    
    # 4. Sorting by product and date (and store)
    df = df.sort_values(by=['store_id', 'product_id', 'date']).reset_index(drop=True)
    
    # Handle new dataset naming conventions
    if 'units_sold' in df.columns and 'sales' not in df.columns:
        df = df.rename(columns={'units_sold': 'sales'})
        
    if 'safety_stock' not in df.columns:
        df['safety_stock'] = 20
    if 'lead_time_days' not in df.columns:
        df['lead_time_days'] = 2
    
    # 5. Missing-value detection & handling
    # Fill sales with 0 if missing. Fill price and other numeric cols with median
    if 'sales' in df.columns:
        df['sales'] = df['sales'].fillna(0)
        
    if 'promotion' not in df.columns:
        if 'Discount_Applied' in df.columns:
            df['promotion'] = (df['Discount_Applied'] > 0).astype(int)
        else:
            df['promotion'] = 0
            
    if 'price' not in df.columns:
        df['price'] = 10.0
    
    for col in ['price', 'promotion']:
        if col in df.columns:
            df[col] = df.groupby('product_id')[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(0)
            
    # 6. Outlier detection (basic IQR method on sales, but keeping them, just flagging)
    if 'sales' in df.columns:
        Q1 = df.groupby(['store_id', 'product_id'])['sales'].transform(lambda x: x.quantile(0.25))
        Q3 = df.groupby(['store_id', 'product_id'])['sales'].transform(lambda x: x.quantile(0.75))
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        # We don't remove outliers because promotions could cause spikes, just flag them
        df['is_outlier'] = (df['sales'] > upper_bound).astype(int)
        
    return df

def fill_missing_dates(df):
    """
    Ensure every store-product combination has a continuous daily time series.
    """
    df = df.copy()
    
    # Get the global date range from the data
    min_date = df['date'].min()
    max_date = df['date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Create complete combinations of store, product and date
    stores = df['store_id'].unique()
    products = df['product_id'].unique()
    
    idx = pd.MultiIndex.from_product([stores, products, all_dates], names=['store_id', 'product_id', 'date'])
    df_complete = pd.DataFrame(index=idx).reset_index()
    
    # Merge with original data
    df = pd.merge(df_complete, df, on=['store_id', 'product_id', 'date'], how='left')
    
    # Fill missing values created by missing dates
    df['sales'] = df['sales'].fillna(0)
    df['promotion'] = df['promotion'].fillna(0)
    df['is_outlier'] = df['is_outlier'].fillna(0)
    
    # Forward fill static features like category, price, safety_stock, lead_time
    cols_to_ffill = ['category', 'price', 'safety_stock', 'lead_time_days']
    for col in cols_to_ffill:
        if col in df.columns:
            df[col] = df.groupby(['store_id', 'product_id'])[col].transform(lambda x: x.ffill().bfill())
            
    return df.sort_values(by=['store_id', 'product_id', 'date']).reset_index(drop=True)

def preprocess_pipeline(file_path):
    df = pd.read_csv(file_path)
    df = clean_data(df)
    df = fill_missing_dates(df)
    return df
