import numpy as np

def engineer_features(df):
    """Add engineered features to the DataFrame."""
    # Liquidity ratio
    df['Liquidity'] = df['24h_volume'] / (df['mkt_cap'])
    # Handle missing values for numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    # Log transformations
    df['log_mkt_cap'] = np.log1p(df['mkt_cap'])
    df['log_24h_volume'] = np.log1p(df['24h_volume'])
    return df