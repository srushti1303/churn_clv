import pandas as pd
import numpy as np

def create_customer_features(customers_df, orders_df=None):
    df = customers_df.copy()
    # simple features from customer_summary
    if 'num_orders' in df.columns:
        df['avg_order_value'] = df['total_order_amount'] / df['num_orders'].replace(0, np.nan)
        df['avg_order_value'] = df['avg_order_value'].fillna(0)

    # ratio features
    if 'total_returns' in df.columns and 'num_orders' in df.columns:
        df['return_rate'] = df['total_returns'] / df['num_orders'].replace(0, np.nan)
        df['return_rate'] = df['return_rate'].fillna(0)

    # interactions features
    if 'num_interactions' in df.columns:
        df['interaction_per_order'] = df['num_interactions'] / df['num_orders'].replace(0, np.nan)
        df['interaction_per_order'] = df['interaction_per_order'].fillna(0)

    if orders_df is not None:
        last_order = orders_df.groupby('customer_id')['order_date'].max().rename('last_order_date')
        df = df.merge(last_order, left_on='customer_id', right_index=True, how='left')
    return df
