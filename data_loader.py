import pandas as pd
from .config import CUSTOMER_CSV, ORDER_CSV

def load_customers(path=CUSTOMER_CSV):
    df = pd.read_csv(path, parse_dates=['signup_date'], low_memory=False)
    # ensure id column type
    df['customer_id'] = df['customer_id'].astype(str)
    return df

def load_orders(path=ORDER_CSV):
    df = pd.read_csv(path, parse_dates=['order_date','signup_date','return_date'], low_memory=False)
    df['customer_id'] = df['customer_id'].astype(str)
    df['order_id'] = df['order_id'].astype(str)
    return df