"""Feature engineering utilities for the exit‑velo project.

All helpers are pure functions that take a pandas DataFrame and return a *copy*
with additional engineered columns so we avoid side effects.

"""
from __future__ import annotations

import pandas as pd
import numpy as np

###############################################################################
# Helper functions
###############################################################################
def _classify_hitters_by_season(
    df: pd.DataFrame,
    batter_col: str = "batter_id",
    season_col: str = "season",
    outcome_col: str = "outcome",
    power_pct: float = 0.8,
) -> pd.Series:
    """
    Season-by-season classification: top `power_pct` hitters by triple+HR rate
    are 'POWER', all others 'CONTACT'. Batters with no hits default to CONTACT.
    """
    # 1) restrict to actual base hits
    hits = df[df[outcome_col].isin(["SINGLE","DOUBLE","TRIPLE","HOME RUN"])].copy()

    # 2) count per (season, batter)
    counts = (
        hits
        .assign(
            contact = lambda d: d[outcome_col].isin(["SINGLE","DOUBLE"]).astype(int),
            power   = lambda d: d[outcome_col].isin(["TRIPLE","HOME RUN"]).astype(int),
        )
        .groupby([season_col, batter_col])
        .agg(
            total_hits   = (outcome_col, "size"),
            contact_hits = ("contact",    "sum"),
            power_hits   = ("power",      "sum"),
        )
    )
    counts["power_rate"]   = counts["power_hits"]   / counts["total_hits"]
    counts["contact_rate"] = counts["contact_hits"] / counts["total_hits"]

    # 3) find the season‐level 80th percentile of power_rate
    pct80 = counts.groupby(level=0)["power_rate"].quantile(power_pct)
    # map it back onto each row
    counts["season_power_80"] = counts.index.get_level_values(season_col).map(pct80)

    # 4) label
    def _label(row):
        return "POWER" if row.power_rate >= row.season_power_80 else "CONTACT"

    labels = counts.apply(_label, axis=1)
    labels.name = "hitter_type"
    return labels




def _rolling_stat_lagged(
    df: pd.DataFrame,
    group_cols: list[str],
    target: str,
    stat: str = "mean",
    window: int = 50,
) -> pd.Series:
    """
    Group-wise rolling statistic using only *previous* rows.
    For each group defined by group_cols, shift the target by 1 row 
    then compute a rolling(window) agg(stat) with min_periods=10.
    """
    # 1) Within each group, shift the target by one so we only use past data
    def shifted_rolling(x: pd.Series) -> pd.Series:
        return x.shift(1).rolling(window=window, min_periods=10).agg(stat)

    rolled = (
        df
        .groupby(group_cols)[target]     # group by batter_id or pitcher_id
        .apply(shifted_rolling)          # shift & roll inside each group
        .reset_index(level=group_cols, drop=True)  # get back a plain Series
    )
    return rolled


###############################################################################
# Public API
###############################################################################

def feature_engineer(df: pd.DataFrame, copy: bool = True) -> pd.DataFrame:
    """Return a DataFrame enriched with engineered features (no leakage)."""

    if copy:
        df = df.copy()

    # 1) Uppercase strings
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.upper())

    # 2) Age
    df["age_sq"]   = df["age"] ** 2
    df["age_bin"]  = pd.qcut(df["age"], q=4, duplicates='drop')

    # 3) Height
    avg_height    = df["batter_height"].mean()
    df["height_diff"] = df["batter_height"] - avg_height

    # 4) Launch & spray bins
    df["la_bin"]    = pd.qcut(df["launch_angle"], q=5, duplicates='drop')
    df["spray_bin"] = pd.qcut(df["spray_angle"], q=3, duplicates='drop')

    # 5) Handedness & matchups
    df["same_hand"]        = (df["batter_hand"] == df["pitcher_hand"])
    df["hand_match"]       = df["batter_hand"] + "_VS_" + df["pitcher_hand"]
    df["pitch_hand_match"] = df["pitch_group"] + "_" + df["hand_match"]

    # 6) Batter lagged rolling EV stats
    df["player_ev_mean50"] = _rolling_stat_lagged(df, ["batter_id"], "exit_velo", "mean", 50)
    df["player_ev_std50"]  = _rolling_stat_lagged(df, ["batter_id"], "exit_velo",  "std", 50)
    gm = df["exit_velo"].mean(); gs = df["exit_velo"].std()
    df["player_ev_mean50"].fillna(gm)
    df["player_ev_std50"].fillna(gs)

    # 7) Pitcher lagged mean
    df["pitcher_ev_mean50"] = _rolling_stat_lagged(df, ["pitcher_id"], "exit_velo", "mean", 50)
    df["pitcher_ev_mean50"].fillna(gm)

    # 8) Center covariates
    df["age_centered"]    = df["age"]    - df["age"].median()
    df["season_centered"] = df["season"] - df["season"].median()
    df["level_idx"]       = df["level_abbr"].map({"AA":0, "AAA":1, "MLB":2})

    # --- 9) New season‐by‐season batter classification ---
    labels = _classify_hitters_by_season(df)
    # bring labels into df by season & batter_id
    labels_df = labels.reset_index()  # columns: [season, batter_id, hitter_type]
    df = df.merge(labels_df, on=["season","batter_id"], how="left")

    # 10) fill anyone still NaN → they had no base hits
    df["hitter_type"] = df["hitter_type"].fillna("CONTACT")
    return df

###############################################################################
# CLI entry‑point (quick smoke test)
###############################################################################

if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_and_clean_data

    raw_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
    df = load_and_clean_data(raw_path, debug = True)
    print(df.head())
    print(df.columns)

    df_fe = feature_engineer(df)

    print("Raw →", df.shape, "//  Feature‑engineered →", df_fe.shape)
    print(df_fe.head())
    print(df_fe.columns)


    # --- DEBUG: batters with no classification ---
    missing_mask = df_fe["hitter_type"].isna()
    missing_df   = df_fe[missing_mask].copy()
    unique_b     = missing_df["batter_id"].nunique()
    print(f"[DEBUG] {unique_b} unique batters have no label (NaN in hitter_type).")

    # --- Fix sort_values key to map each array's length ---
    outcome_by_batter = (
        missing_df
        .groupby("batter_id")["outcome"]
        .unique()
        .sort_values(
            key=lambda s: s.map(len),      # for each entry, use len(array)
            ascending=False
        )
    )
    print("[DEBUG] Sample of missing batters → their unique outcomes:")
    print(outcome_by_batter.head(10).to_dict())

    print("[DEBUG] Outcome value counts among missing batters:")
    print(missing_df["outcome"].value_counts().to_dict())
