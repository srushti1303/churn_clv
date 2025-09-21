import pandas as pd
import numpy as np

def load_data():
    customers = pd.read_csv("data/customers_dim.csv")
    orders = pd.read_csv("data/orders_fact.csv")
    order_items = pd.read_csv("data/order_items.csv")
    products = pd.read_csv("data/products_dim.csv")
    interactions = pd.read_csv("data/customer_interactions.csv")
    marketing = pd.read_csv("data/marketing_spend.csv")
    returns = pd.read_csv("data/returns_refunds.csv")
    return customers, orders, order_items, products, interactions, marketing, returns

def summarize_per_order(customers, orders, order_items, products, interactions, marketing, returns):

    # ---- Orders + Customers ----
    order_df = pd.merge(orders, customers, on="customer_id", how="left")

    # ---- Items + Products aggregated per order ----
    items_df = pd.merge(order_items, products, on="product_id", how="left")
    items_summary = items_df.groupby("order_id").agg({
        "order_item_id": "count",                          # num order items
        "product_id": lambda x: ','.join(map(str, set(x))),# distinct product_ids
        "quantity": "sum",                                 # total quantity
        "price_x": "sum",                                  # sum of item prices
        "category": lambda x: ','.join(set(x.dropna())),
        "subcategory": lambda x: ','.join(set(x.dropna()))
    }).reset_index().rename(columns={
        "order_item_id": "num_items",
        "product_id": "products_in_order",
        "quantity": "total_quantity",
        "price_x": "total_item_price",
        "category": "categories",
        "subcategory": "subcategories"
    })

    # ---- Returns per order ----
    returns_summary = returns.groupby("order_id").agg({
        "return_id": "count",
        "return_date": lambda x: ','.join(set(x.dropna().astype(str))),
        "reason": lambda x: ','.join(set(x.dropna()))
    }).reset_index().rename(columns={
        "return_id": "num_returns",
        "reason": "return_reasons"
    })

    # ---- Interactions per customer ----
    inter_summary = interactions.groupby("customer_id").agg({
        "interaction_id": "count",
        "type": lambda x: ','.join(set(x.dropna())),
        "text": lambda x: ' | '.join(set(x.dropna()))
    }).reset_index().rename(columns={
        "interaction_id": "num_interactions",
        "type": "interaction_types",
        "text": "interaction_texts"
    })

    # ---- Marketing per acquisition channel ----
    marketing_summary = marketing.groupby("channel").agg({
        "campaign_id": "count",
        "spend": "sum",
        "impressions": "sum",
        "clicks": "sum"
    }).reset_index().rename(columns={
        "campaign_id": "num_campaigns",
        "spend": "total_spend",
        "impressions": "total_impressions",
        "clicks": "total_clicks"
    })

    # ---- Combine everything ----
    order_summary = (
        order_df
        .merge(items_summary, on="order_id", how="left")
        .merge(returns_summary, on="order_id", how="left")
        .merge(inter_summary, on="customer_id", how="left")
        .merge(marketing_summary, left_on="acquisition_channel", right_on="channel", how="left")
    )


    # Numeric columns: fill with median (or mean if you prefer)
    for col in ['num_returns', 'num_interactions', 'num_campaigns', 
                'total_spend', 'total_impressions', 'total_clicks']:
        order_summary[col] = order_summary[col].fillna(order_summary[col].median())

    # Date columns: fill with most frequent date (mode)
    order_summary['return_date'] = order_summary['return_date'].fillna(order_summary['return_date'].mode()[0])

    # Categorical/text columns: fill with mode (most common value)
    for col in ['return_reasons', 'interaction_types', 'interaction_texts', 'channel']:
        order_summary[col] = order_summary[col].fillna(order_summary[col].mode()[0])

    print(order_summary.isnull().sum())
    return order_summary


def summarize_per_customer(customers, orders, order_items, interactions, marketing, returns):
    # ---- Orders aggregated per customer ----
    orders_summary = orders.groupby("customer_id").agg({
        "order_id": "count",
        "order_amount": "sum",
        "discount_amount": "sum"
    }).reset_index().rename(columns={
        "order_id": "num_orders",
        "order_amount": "total_order_amount",
        "discount_amount": "total_discounts"
    })

    # ---- Items aggregated per customer ----
    order_items_link = pd.merge(order_items, orders[["order_id","customer_id"]], on="order_id", how="left")
    items_summary = order_items_link.groupby("customer_id").agg({
        "order_item_id": "count",
        "product_id": lambda x: ','.join(map(str, set(x))),
        "quantity": "sum",
        "price": "sum"
    }).reset_index().rename(columns={
        "order_item_id": "total_items",
        "product_id": "unique_products",
        "quantity": "total_quantity",
        "price": "total_item_price"
    })

    # ---- Returns aggregated per customer ----
    returns_link = pd.merge(returns, orders[["order_id","customer_id"]], on="order_id", how="left")
    returns_summary = returns_link.groupby("customer_id").agg({
        "return_id": "count",
        "reason": lambda x: ','.join(set(x.dropna()))
    }).reset_index().rename(columns={
        "return_id": "total_returns",
        "reason": "return_reasons"
    })

    # ---- Interactions per customer ----
    inter_summary = interactions.groupby("customer_id").agg({
        "interaction_id": "count",
        "type": lambda x: ','.join(set(x.dropna())),
        "text": lambda x: ' | '.join(set(x.dropna()))
    }).reset_index().rename(columns={
        "interaction_id": "num_interactions",
        "type": "interaction_types",
        "text": "interaction_texts"
    })

    # ---- Marketing exposure per channel ----
    marketing_summary = marketing.groupby("channel").agg({
        "campaign_id": "count",
        "spend": "sum",
        "impressions": "sum",
        "clicks": "sum"
    }).reset_index().rename(columns={
        "campaign_id": "num_campaigns",
        "spend": "total_spend",
        "impressions": "total_impressions",
        "clicks": "total_clicks"
    })

    # ---- Combine all into customer-level dataset ----
    customer_summary = (
        customers
        .merge(orders_summary, on="customer_id", how="left")
        .merge(items_summary, on="customer_id", how="left")
        .merge(returns_summary, on="customer_id", how="left")
        .merge(inter_summary, on="customer_id", how="left")
        .merge(marketing_summary, left_on="acquisition_channel", right_on="channel", how="left")
    )

    def count_products(x):
        if pd.isna(x):
            return np.nan
        return len(str(x).split(','))

    customer_summary['unique_products'] = customer_summary['unique_products'].apply(count_products)
    customer_summary['unique_products'] = pd.to_numeric(customer_summary['unique_products'], errors='coerce')

    # ---- Fill nulls ----
    # Fill numeric columns with median
    numeric_cols = [
        'num_orders', 'total_order_amount', 'total_discounts', 'total_items', 'unique_products',
        'total_quantity', 'total_item_price', 'total_returns', 'num_interactions', 'num_campaigns',
        'total_spend', 'total_impressions', 'total_clicks'
    ]
    for col in numeric_cols:
        customer_summary[col] = customer_summary[col].fillna(customer_summary[col].median())

    # Fill categorical/text columns with mode (most frequent value)
    mode_cols = ['return_reasons', 'interaction_types', 'interaction_texts', 'channel']
    for col in mode_cols:
        customer_summary[col] = customer_summary[col].fillna(customer_summary[col].mode()[0])
        
    print(customer_summary.isnull().sum())
    return customer_summary

if __name__ == "__main__":
    customers, orders, order_items, products, interactions, marketing, returns = load_data()
    order_summary = summarize_per_order(customers, orders, order_items, products, interactions, marketing, returns)
    customer_summary = summarize_per_customer(customers, orders, order_items, interactions, marketing, returns)

    order_summary.to_csv("data/order_summary.csv", index=False)
    customer_summary.to_csv("data/customer_summary.csv", index=False)
