REQUIRED_COLUMNS = ['date', 'product_id', 'sales']
OPTIONAL_COLUMNS = ['store_id', 'price', 'promotion', 'category']

# Default ecommerce to standard metric mapping
ECOMMERCE_MAPPING = {
    'Date': 'date',
    'Location': 'store_id',
    'Product_Category': 'category',
    'Product_Name': 'product_id',
    'Unit_Price': 'price',
    'Quantity': 'sales',
    'units_sold': 'sales'
}
