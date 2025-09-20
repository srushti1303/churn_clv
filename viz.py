import plotly.express as px

def plot_revenue_over_time(df_time):
    """
    df_time: DataFrame with date/time column as first column and value column as second
    """
    x_col = df_time.columns[0]
    y_col = df_time.columns[1]
    fig = px.line(df_time, x=x_col, y=y_col, title="Revenue Over Time")
    return fig

def plot_top_categories(cat_rev_df):
    fig = px.bar(cat_rev_df, x='categories', y='order_amount', title='Top Categories by Revenue')
    return fig

def plot_cohort_counts(cohort_df):
    fig = px.bar(cohort_df, x=cohort_df.columns[0], y=cohort_df.columns[1], title='Cohort Counts')
    return fig
