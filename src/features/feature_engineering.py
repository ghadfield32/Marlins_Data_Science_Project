"""Feature engineering utilities for the exit‑velo project.

All helpers are pure functions that take a pandas DataFrame and return a *copy*
with additional engineered columns so we avoid side effects.

Example
-------
>>> from src.data.load_data import load_raw
>>> from src.features.feature_engineering import feature_engineer
>>> df = load_raw()
>>> df_fe = feature_engineer(df)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

###############################################################################
# Helper functions
###############################################################################

def _rolling_stat(
    df: pd.DataFrame,
    group_cols: list[str],
    target: str,
    stat: str = "mean",
    window: int = 50,
) -> pd.Series:
    """Group‑wise rolling statistic.  Sorted by season order of appearance.

    The function first sorts by the index order (assumed chronological inside each
    group) then applies a rolling window with *min_periods=10* so early samples
    are not overly noisy.
    """
    return (
        df.sort_values(group_cols)
        .groupby(group_cols)[target]
        .rolling(window, min_periods=10)
        .agg(stat)
        .reset_index(level=group_cols, drop=True)
    )

###############################################################################
# Public API
###############################################################################

def feature_engineer(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Return a DataFrame enriched with engineered features.

    Parameters
    ----------
    df   : raw data as returned by ``load_raw``
    copy : if *True* (default) operate on a copy so the original is untouched.
    """

    if copy:
        df = df.copy()

    # ────────────────────────────────────────────────────────────────────────
    # 1. Basic type harmonisation & canonical casing
    # ────────────────────────────────────────────────────────────────────────
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.upper())

    # ────────────────────────────────────────────────────────────────────────
    # 2. Age engineering (dynamic quantile bins)
    # ────────────────────────────────────────────────────────────────────────
    df["age_sq"] = df["age"] ** 2  # capture non‑linear aging curve
    n_age_bins = 4
    df["age_bin"] = pd.qcut(df["age"], q=n_age_bins, duplicates='drop')
    
    # ────────────────────────────────────────────────────────────────────────
    # 3. Height normalisation relative to MLB average (~74 inches)
    # ────────────────────────────────────────────────────────────────────────
    avg_batter_height = df["batter_height"].mean()
    
    # Use the average batter height to calculate height_diff
    df["height_diff"] = df["batter_height"] - avg_batter_height

    # ────────────────────────────────────────────────────────────────────────
    # 4. Launch / spray angle buckets (dynamic quantile bins) & barrel indicator
    # ────────────────────────────────────────────────────────────────────────
    # Dynamic launch-angle bins
    n_la_bins = 5
    df["la_bin"] = pd.qcut(df["launch_angle"], q=n_la_bins, duplicates='drop')

    # Dynamic spray-angle bins
    n_spray_bins = 3
    df["spray_bin"] = pd.qcut(df["spray_angle"], q=n_spray_bins, duplicates='drop')

    # Statcast barrel proxy: EV >= 98 & 26° > LA >= 8°
    df["is_barrel"] = (
        (df["exit_velo"] >= 98) & (df["launch_angle"] >= 8) & (df["launch_angle"] < 26)
    ).astype("category")


    # ────────────────────────────────────────────────────────────────────────
    # 6. Handedness & matchup indicators
    # ────────────────────────────────────────────────────────────────────────
    df["same_hand"] = (df["batter_hand"] == df["pitcher_hand"])
    df["hand_match"] = df["batter_hand"] + "_VS_" + df["pitcher_hand"]

    # ────────────────────────────────────────────────────────────────────────
    # 7. Pitch‑type interactions
    # ────────────────────────────────────────────────────────────────────────
    df["pitch_hand_match"] = df["pitch_group"] + "_" + df["hand_match"]

    # ────────────────────────────────────────────────────────────────────────
    # 8. Player‑level historical stats (rolling EV mean & SD)
    #    These capture each hitter’s *latent* ability and shrink early samples
    #    via wider rolling windows.
    # ────────────────────────────────────────────────────────────────────────
    df["player_ev_mean50"] = _rolling_stat(df, ["batter_id"], "exit_velo", "mean", 50)
    df["player_ev_std50"] = _rolling_stat(df, ["batter_id"], "exit_velo", "std", 50)
    # ✱ NEW: safe assignment, no inplace chain ✱
    ev_mean_global = df["exit_velo"].mean()
    ev_std_global  = df["exit_velo"].std()
    df["player_ev_mean50"] = df["player_ev_mean50"].fillna(ev_mean_global)
    df["player_ev_std50"]  = df["player_ev_std50"].fillna(ev_std_global)

    # 9a. Hard‑hit & barrel‑adjacent flags
    df["hard_hit"] = (df["exit_velo"] >= 95).astype("category")
    df["near_barrel"] = (
        (df["exit_velo"].between(95, 98)) &
        (df["launch_angle"].between(5, 30))
    ).astype("category")

    # 9b. EV × LA and distance proxy
    df["ev_la_product"] = df["exit_velo"] * (df["launch_angle"] + 90)
    df["est_distance"] = df["exit_velo"] * df["hangtime"]
    df["ev_la_sqrt"] = np.sqrt(df["ev_la_product"].clip(lower=0))

    # 10. Pitcher rolling stats 
    pitcher_mean = df["exit_velo"].mean()
    df["pitcher_ev_mean50"] = _rolling_stat(df, ["pitcher_id"], "exit_velo", "mean", 50)
    df["pitcher_ev_mean50"] = df["pitcher_ev_mean50"].fillna(pitcher_mean)


    # 11. Outcome encoding – simple value mapping for power/speed signal.
    _OUTCOME_W = {
        "out": 0,
        "single": 1,
        "double": 2,
        "triple": 3,
        "home run": 4,
    }
    df["outcome_val"] = df["outcome"].str.lower().map(_OUTCOME_W)

    # centred covariates
    df["age_centered"]    = df["age"] - df["age"].median()
    df["season_centered"] = df["season"] - df["season"].median()   # ⬅ NEW
    df["level_idx"]       = df["level_abbr"].map({"AA": 0, "AAA": 1, "MLB": 2})
    
    return df

###############################################################################
# CLI entry‑point (quick smoke test)
###############################################################################

if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_raw

    raw_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
    df = load_raw(raw_path)
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
    df_fe = feature_engineer(df)

    print("Raw →", df.shape, "//  Feature‑engineered →", df_fe.shape)
    print(df_fe.head())
    print(df_fe.columns)
