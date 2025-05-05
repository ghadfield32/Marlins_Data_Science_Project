import pandas as pd


def load_raw(path='data/Research Data Project/Research Data Project/exit_velo_project_data.csv'):
    """
    Load raw exit velocity data from the data directory.
    
    Returns:
        pd.DataFrame: Raw exit velocity data
    """
    # This is a placeholder - in a real implementation, you would load actual data
    # For example: df = pd.read_csv('data/raw/exit_velo.csv')

    df = pd.read_csv(path)
    return df 


if __name__ == "__main__":
    path = 'data/Research Data Project/Research Data Project/exit_velo_project_data.csv'
    df = load_raw(path)
    print(df.head())
    print(df.columns)

    # --- inspect nulls in the raw data ---
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print("✅  No missing values in raw data.")
    else:
        print("=== Raw data null counts ===")
        for col, cnt in null_counts.items():
            print(f" • {col!r}: {cnt} missing")
    
