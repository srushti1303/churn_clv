import pandas as pd
import plotly.express as px

def orders_time_summary(orders_df, freq='D'):
    orders_df = orders_df.copy()
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
    series = orders_df.set_index('order_date').resample(freq)['order_amount'].sum().reset_index()
    return series

def cohort_retention_table(customers_df, orders_df, cohort_col='cohort'):
    # Simple cohort counts by cohort and month of first purchase
    customers_df = customers_df.copy()
    orders_df = orders_df.copy()
    customers_df['signup_month'] = pd.to_datetime(customers_df['signup_date']).dt.to_period('M')
    cohort_counts = customers_df.groupby('signup_month').size().reset_index(name='cohort_size')
    return cohort_counts

def top_n_categories(orders_df, n=10):
    if 'categories' not in orders_df.columns:
        return None
    cat_rev = orders_df.groupby('categories')['order_amount'].sum().reset_index().sort_values('order_amount', ascending=False).head(n)
    return cat_rev

def returns_by_reason(orders_df, top_n=10):
    if 'return_reasons' not in orders_df.columns:
        return None
    reasons = orders_df['return_reasons'].astype(str).str.split(';').explode().value_counts().reset_index()
    reasons.columns = ['reason','count']
    return reasons.head(top_n)