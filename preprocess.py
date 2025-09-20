import pandas as pd
import numpy as np
from .config import PROCESSED_DIR
import os

def clean_customers(df):
    df = df.copy()
    # fill missing numeric fields with 0
    numeric_cols = ['num_orders','total_order_amount','total_discounts','total_items','unique_products',
                    'total_quantity','total_item_price','total_returns','num_interactions','num_campaigns',
                    'total_spend','total_impressions','total_clicks']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # normalize strings
    df['gender'] = df.get('gender', pd.Series()).fillna('Unknown').astype(str)
    df['loyalty_status'] = df.get('loyalty_status', pd.Series()).fillna('None').astype(str)
    return df

def clean_orders(df):
    df = df.copy()
    # numeric conversions
    numcols = ['order_amount','discount_amount','num_items','total_quantity','total_item_price','num_returns']
    for c in numcols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # trim categories/subcategories
    for c in ['categories','subcategories']:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str)

    return df

def save_parquet(df, path):
    os.makedirs(path.parent, exist_ok=True)
    df.to_parquet(path, index=False)

def process_and_save(customers_df, orders_df):
    cust_clean = clean_customers(customers_df)
    ord_clean = clean_orders(orders_df)
    save_parquet(cust_clean, PROCESSED_DIR / "customers_clean.parquet")
    save_parquet(ord_clean, PROCESSED_DIR / "orders_clean.parquet")
    return cust_clean, ord_clean
