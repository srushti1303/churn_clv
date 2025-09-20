from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

def prepare_summary(orders_df, monetary_value_col='order_amount'):
    """
    Create the lifetimes summary table with columns:
    - frequency: number of repeat purchases (frequency)
    - recency: recency (in same units as T)
    - T: customer age (observation period length)
    - monetary_value: average order value per customer
    Returns the summary DataFrame.
    """
    orders = orders_df.copy()
    orders['order_date'] = pd.to_datetime(orders['order_date'])

    snapshot_date = orders['order_date'].max() + pd.Timedelta(days=1)

    summary = summary_data_from_transaction_data(
        transactions=orders,
        customer_id_col='customer_id',
        datetime_col='order_date',
        monetary_value_col=monetary_value_col,
        observation_period_end=snapshot_date
    )

    # summary has columns frequency, recency, T, monetary_value
    return summary

def train_bgfgg(orders_df, monetary_value_col='order_amount', return_summary=False):
    """
    Fit BetaGeoFitter and GammaGammaFitter on the provided orders DataFrame.
    Returns bgf, ggf and optionally the summary used for fitting.
    """
    summary = prepare_summary(orders_df, monetary_value_col=monetary_value_col)
    # Filter out invalid rows
    summary = summary[(summary['monetary_value'] > 0) & (summary['frequency'] > 0)]

    # Optional: cap extreme monetary values to reduce variance
    summary['monetary_value'] = summary['monetary_value'].clip(upper=summary['monetary_value'].quantile(0.99))

    bgf = BetaGeoFitter(penalizer_coef=1.0)
    bgf.fit(summary['frequency'], summary['recency'], summary['T'], penalizer_coef=1)
    
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    try:
        ggf.fit(summary['frequency'], summary['monetary_value'])
    except Exception as e:
        print(f"Gamma-Gamma failed to converge: {e}")
        ggf = None  # or return only bgf if needed

    print("BG/NBD and Gamma-Gamma fit complete.")
    if return_summary:
        return bgf, ggf, summary
    return bgf, ggf

if __name__ == "__main__":
    from src.data_loader import load_orders
    orders = load_orders()
    bgf, ggf, summary = train_bgfgg(orders, return_summary=True)
    print(summary.head())
