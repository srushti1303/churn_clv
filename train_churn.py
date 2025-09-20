# Train a churn classifier (XGBoost). Produces churn_next_30d as label via time-split.
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

def make_label(orders, label_days=30):
    """
    Create churn label using a time-based holdout:
    - snapshot_date = max(order_date)
    - cutoff_date = snapshot_date - label_days
    For each customer:
      X data = orders with order_date <= cutoff_date
      Label = 1 if customer placed any order in (cutoff_date, snapshot_date] else 0
    Returns: DataFrame with columns ['customer_id', 'label_cutoff', 'churn_next_30d']
    """
    orders = orders.copy()
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    snapshot_date = orders['order_date'].max()
    cutoff_date = snapshot_date - pd.Timedelta(days=label_days)

    # customers with any orders after cutoff
    future_orders = orders[orders['order_date'] > cutoff_date]
    customers_future = set(future_orders['customer_id'].astype(str).unique())

    # label: churn_next_30d = 0 if they had order in next 30d, else 1
    all_customers = orders['customer_id'].astype(str).unique()
    rows = []
    for cid in all_customers:
        label = 0 if cid in customers_future else 1
        rows.append({'customer_id': str(cid), 'churn_next_30d': label})

    label_df = pd.DataFrame(rows)
    return label_df, cutoff_date, snapshot_date

def build_features(customers_df, orders_df, cutoff_date):
    """
    Create features using only orders up to cutoff_date (to avoid label leakage).
    Uses customer_summary fields and order-aggregations.
    """
    customers_df = customers_df.copy()
    orders_df = orders_df.copy()
    orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])

    orders_train = orders_df[orders_df['order_date'] <= cutoff_date]

    # order-level aggregations per customer (train window)
    agg = orders_train.groupby('customer_id').agg(
        train_num_orders=('order_id', 'nunique'),
        train_total_sales=('order_amount', 'sum'),
        train_avg_order_value=('order_amount', 'mean'),
        train_num_returns=('num_returns', 'sum'),
        train_total_quantity=('total_quantity', 'sum'),
    ).reset_index()

    # last_order_date and recency (days between last order and cutoff_date)
    last_order = orders_train.groupby('customer_id')['order_date'].max().reset_index().rename(columns={'order_date':'last_order_date'})
    last_order['recency_days'] = (cutoff_date - last_order['last_order_date']).dt.days

    feat = customers_df.merge(agg, left_on='customer_id', right_on='customer_id', how='left')
    feat = feat.merge(last_order[['customer_id','recency_days']], on='customer_id', how='left')

    # fill NaNs with zeros where meaningful
    numcols = ['train_num_orders','train_total_sales','train_avg_order_value','train_num_returns','train_total_quantity','recency_days']
    for c in numcols:
        if c in feat.columns:
            feat[c] = feat[c].fillna(0)

    # some ratios
    feat['returns_per_order'] = feat['train_num_returns'] / feat['train_num_orders'].replace(0, np.nan)
    feat['returns_per_order'] = feat['returns_per_order'].fillna(0)

    # encode simple categorical columns with get_dummies (small)
    cat_cols = []
    for candidate in ['gender','cohort','acquisition_channel','loyalty_status']:
        if candidate in feat.columns:
            cat_cols.append(candidate)
    if cat_cols:
        feat = pd.get_dummies(feat, columns=cat_cols, drop_first=True)

    # drop columns that are identifiers or text long fields
    drop_cols = ['signup_date','return_reasons','interaction_types','interaction_texts','channel']
    for c in drop_cols:
        if c in feat.columns:
            feat = feat.drop(columns=[c])

    # ensure customer_id present
    feat['customer_id'] = feat['customer_id'].astype(str)

    return feat

def train_churn(customers_df, orders_df, label_days=30, return_eval=False):
    """
    Trains an XGBoost churn classifier. Does NOT save the model to disk by default.
    If return_eval True: returns (model, X_val, y_val, metrics_dict)
    """
    # create label
    labels_df, cutoff_date, snapshot_date = make_label(orders_df, label_days=label_days)

    # build features using data up to cutoff_date
    feat = build_features(customers_df, orders_df, cutoff_date)

    # join label
    data = feat.merge(labels_df, on='customer_id', how='left')
    # If some customers in customer_summary but no orders at all, consider them churned (label 1)
    data['churn_next_30d'] = data['churn_next_30d'].fillna(1).astype(int)

    # prepare X/y
    X = data.drop(columns=['customer_id','churn_next_30d'], errors='ignore')
    y = data['churn_next_30d'].astype(int)

    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # xgboost DMatrix
    for col in X_train.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
    for col in X_val.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X_val[col] = le.fit_transform(X_val[col].astype(str))

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'eta': 0.05,
        'max_depth': 6,
        'seed': 42,
    }

    watchlist = [(dtrain, 'train'), (dval, 'eval')]
    bst = xgb.train(params, dtrain, num_boost_round=300, evals=watchlist, early_stopping_rounds=20, verbose_eval=False)

    # eval
    preds = bst.predict(dval)
    auc = roc_auc_score(y_val, preds)
    yhat = (preds > 0.5).astype(int)
    cm = confusion_matrix(y_val, yhat)
    report = classification_report(y_val, yhat, output_dict=True)

    metrics = {
        'auc': float(auc),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }

    print("Churn train complete. Validation AUC:", auc)
    print("Classification report (val):")
    print(classification_report(y_val, yhat))

    if return_eval:
        # return model and some evaluation objects for app use
        return bst, X_val, y_val, metrics
    else:
        return bst

# If run as script
if __name__ == "__main__":
    import json
    from ..data_loader import load_customers, load_orders
    customers = load_customers()
    orders = load_orders()
    model, X_val, y_val, metrics = train_churn(customers, orders, return_eval=True)
    print("Metrics:")
    print(json.dumps(metrics, indent=2))
