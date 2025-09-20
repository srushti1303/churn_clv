# notebooks/01_exploratory_orders.py
# %% [markdown]
# # Exploratory analysis — Orders (Operational / Logistics)

# Steps:
# 1. Load order_summary.csv
# 2. Summary KPIs: total orders, revenue, AOV, returns %
# 3. Time series: revenue & order count
# 4. Order status and fulfillment analysis
# 5. Returns analysis and top return reasons
#
# %% 
import pandas as pd
from pathlib import Path
import plotly.express as px

ROOT = Path(__file__).resolve().parents[1]
ORD_CSV = ROOT / "data" / "raw" / "order_summary.csv"

orders = pd.read_csv(ORD_CSV, parse_dates=['order_date','signup_date','return_date'], low_memory=False)
orders['order_amount'] = pd.to_numeric(orders['order_amount'])
orders['num_items'] = pd.to_numeric(orders['num_items'], errors='coerce').fillna(0)

# %% 
# KPIs
total_orders = len(orders)
total_revenue = orders['order_amount'].sum()
aov = orders['order_amount'].mean()
total_returns = orders['num_returns'].sum() if 'num_returns' in orders.columns else 0

print("Total orders:", total_orders)
print("Total revenue:", total_revenue)
print("AOV:", aov)
print("Total returns:", total_returns)

# %%
# Time series revenue
orders_by_day = orders.set_index('order_date').resample('D')['order_amount'].sum().reset_index()
fig = px.line(orders_by_day, x='order_date', y='order_amount', title="Revenue by Day")
fig.show()

# %%
# Order status breakdown
if 'order_status' in orders.columns:
    fig2 = px.histogram(orders, x='order_status', title='Order status distribution')
    fig2.show()

# %%
# Top categories
if 'categories' in orders.columns:
    cat_rev = orders.groupby('categories')['order_amount'].sum().reset_index().sort_values('order_amount', ascending=False).head(20)
    fig3 = px.bar(cat_rev, x='categories', y='order_amount', title='Top categories by revenue')
    fig3.show()

# %%
# Returns reasons
if 'return_reasons' in orders.columns:
    reasons = orders['return_reasons'].astype(str).str.split(';').explode().value_counts().reset_index()
    reasons.columns = ['reason','count']
    fig4 = px.bar(reasons.head(15), x='reason', y='count', title='Top return reasons')
    fig4.show()

# %%
