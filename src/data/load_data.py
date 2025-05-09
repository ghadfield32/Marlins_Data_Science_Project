import pandas as pd
import numpy as np

def load_data(path='data/Research Data Project/Research Data Project/exit_velo_project_data.csv'):
    """
    Load raw exit velocity data from the data directory.

    Returns:
        pd.DataFrame: Raw exit velocity data
    """

    df = pd.read_csv(path)
    return df 



# ──  utility ─────────────────────────────────────────────────────────
def clean_raw(df: pd.DataFrame, 
              target: str = "exit_velo"
              ,debug: bool = False) -> pd.DataFrame:
    """
    Central place to:
      1. Drop rows with NaN in *target*.
      2. Print a concise null‑summary afterwards (one line per column).

    Returns a *fresh copy* (never mutates in‑place).
    """
    if debug:
        print(f"Dropping rows with NaN in {target}")
        print(df.isna().sum())

    drop_columns = [target, "hit_type", "launch_angle"]
    out = df.dropna(subset=drop_columns).copy()   

    if debug:
        print(f"after rows dropped with NaN in {target}")
        print(out.isna().sum())
    # quick dashboard
    nulls = out.isna().sum()
    non_zero = nulls[nulls > 0]
    if non_zero.empty:
        print(f"✅  After target‑filter, no other nulls (n={len(out):,}).")
    else:
        print("⚠️  Nulls after target‑filter:")
        for col, cnt in non_zero.items():
            pct = cnt / len(out)
            print(f"  • {col:<15} {cnt:>7,} ({pct:5.2%})")

    return out




def load_and_clean_data(path='data/Research Data Project/Research Data Project/exit_velo_project_data.csv'
                        ,debug: bool = False):
    df = load_data(path)
    df = clean_raw(df, debug = True) 
    return df


if __name__ == "__main__":
    path = 'data/Research Data Project/Research Data Project/exit_velo_project_data.csv'

    df = load_and_clean_data(path, debug = True)
    print(df.head())
    print(df.columns)

