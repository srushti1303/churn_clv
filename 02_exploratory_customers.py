# notebooks/02_exploratory_customers.py
# %% [markdown]
# # Exploratory analysis — Customers (Churn / Segmentation / LTV)

# Steps:
# 1. Load customer_summary.csv and order_summary.csv
# 2. Look at distributions: num_orders, total_order_amount, avg_order_value
# 3. Cohort analysis and retention
# 4. Segmentation (LTV buckets) and channel analysis
#
# %%
import pandas as pd
from pathlib import Path
import plotly.express as px
from lifetimes.utils import summary_data_from_transaction_data

ROOT = Path(__file__).resolve().parents[1]
CUST_CSV = ROOT / "data" / "raw" / "customer_summary.csv"
ORD_CSV = ROOT / "data" / "raw" / "order_summary.csv"

customers = pd.read_csv(CUST_CSV, parse_dates=['signup_date'], low_memory=False)
orders = pd.read_csv(ORD_CSV, parse_dates=['order_date','signup_date','return_date'], low_memory=False) 

# %%
# Basic distributions
print("Num customers:", len(customers))
display(customers[['num_orders','total_order_amount']].describe())

fig = px.histogram(customers, x='num_orders', nbins=50, title='Distribution of num_orders')
fig.show()

fig2 = px.histogram(customers, x='total_order_amount', nbins=50, title='Distribution of total_order_amount')
fig2.show()

# %%
# Cohort counts
if 'cohort' in customers.columns:
    cohort_counts = customers['cohort'].value_counts().reset_index()
    cohort_counts.columns = ['cohort', 'count']
    figc = px.bar(cohort_counts, x='cohort', y='count', title='Cohort counts')
    figc.show()

# %%
# LTV buckets (simple)
customers['avg_order_value'] = customers['total_order_amount'] / customers['num_orders'].replace(0, pd.NA)
# Compute quantile bins manually
quantiles = customers['total_order_amount'].quantile([0, 0.25, 0.5, 0.75, 1.0]).unique()

# Drop duplicate edges
unique_bins = pd.Series(quantiles).drop_duplicates().values

# Create matching labels
labels = ['low', 'mid', 'high', 'very_high'][:len(unique_bins)-1]

# Apply qcut with cleaned bins and labels
customers['ltv_bucket'] = pd.cut(customers['total_order_amount'], bins=unique_bins, labels=labels, include_lowest=True)
fig_ltv = px.box(customers, x='ltv_bucket', y='total_order_amount', title='LTV buckets')
fig_ltv.show()

# %%
# Prepare lifetimes summary table for CLV modeling
summary = summary_data_from_transaction_data(orders, 'customer_id', 'order_date', monetary_value_col='order_amount', observation_period_end=orders['order_date'].max() + pd.Timedelta(days=1))
display(summary.head())

# %%
# simple scatter of frequency vs monetary
fig_sc = px.scatter(summary, x='frequency', y='monetary_value', title='Frequency vs Monetary Value')
fig_sc.show()

# %%
