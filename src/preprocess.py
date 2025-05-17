import pandas as pd

def load_data(filepath, sheet_name=None):
    
    if filepath.endswith('.csv') or filepath.endswith('.xls'):
        return pd.read_excel(filepath, sheet_name=sheet_name)
    else:
        return pd.read_csv(filepath)

def clean_data(df):
    
    if 'date' in df.columns and df['date'].nunique() == 1:
        df = df.drop(columns=['date'])
    # Drop rows with impossible values (example: negative market cap)
    if 'mkt_cap' in df.columns:
        df = df[df['mkt_cap'] >= 0]
    # Fill missing values with median
    df.fillna(df.median(), inplace=True)
    return df