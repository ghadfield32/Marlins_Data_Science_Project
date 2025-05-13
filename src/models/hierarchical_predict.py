

import json
from pathlib import Path

import arviz as az
import pandas as pd
import numpy as np

from src.utils.posterior import align_batter_codes
from src.features.feature_engineering import feature_engineer


# ─── new helper at top of file ────────────────────────────────────────────
def get_top_hitters(
    df: pd.DataFrame,
    hitter_col: str = "hitter_type",
    n: int = 5,
    verbose: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given a DataFrame with predictions and a 'hitter_type' column,
    return two DataFrames of the top-n POWER and CONTACT hitters by pred_mean.
    If the column is missing, returns two empty DataFrames (and prints a warning).
    """
    if hitter_col not in df.columns:
        if verbose:
            print(f"[Warning] Column '{hitter_col}' not found; skipping top-hitter extraction.")
        empty = pd.DataFrame(columns=[ "batter_id", hitter_col, "pred_mean", "pred_lo95", "pred_hi95" ])
        return empty, empty

    # 1) Filter power vs. contact, case‐insensitive
    power_mask   = df[hitter_col].str.contains("POWER",   case=False, na=False)
    contact_mask = df[hitter_col].str.contains("CONTACT", case=False, na=False)

    power_df   = df.loc[power_mask]
    contact_df = df.loc[contact_mask]

    # 2) Take the top-n by pred_mean
    top_power   = power_df.nlargest(n, "pred_mean")[ ["batter_id", hitter_col, "pred_mean", "pred_lo95", "pred_hi95"] ]
    top_contact = contact_df.nlargest(n, "pred_mean")[ ["batter_id", hitter_col, "pred_mean", "pred_lo95", "pred_hi95"] ]

    if verbose:
        print(f"[Top {n}] power hitters:\n{top_power}")
        print(f"[Top {n}] contact hitters:\n{top_contact}")

    return top_power, top_contact



def simplified_prepare_validation(df_val: pd.DataFrame, median_age: float, verbose: bool = False) -> pd.DataFrame:
    """
    Prepare validation dataset with simplified approach, handling missing columns gracefully.

    Assumptions:
    - All predictions are for MLB-level competition (level_idx = 2).
    - Missing age defaults to median training age (age_centered = 0).

    Parameters
    ----------
    df_val : pd.DataFrame
        Raw validation dataframe (must include 'season', 'batter_id', optional 'age').
    median_age : float
        Median age computed from training data for centering.
    verbose : bool
        If True, prints debug information.

    Returns
    -------
    pd.DataFrame
        DataFrame with added 'age_centered' and 'level_idx'.
    """
    df_val = df_val.copy()

    # Default to MLB level for all observations
    df_val['level_idx'] = 2

    # Center age
    if 'age' in df_val.columns:
        df_val['age_centered'] = df_val['age'] - median_age
    else:
        # Using median training age => zero effect
        df_val['age_centered'] = 0.0

    if verbose:
        print(f"[Validation Prep] level_idx set to 2 for all rows, age_centered stats: min={df_val['age_centered'].min()}, max={df_val['age_centered'].max()}")

    return df_val


def predict_from_summaries(
    roster_csv: Path,
    raw_csv:    Path,
    posterior_parquet: Path,
    global_effects_json: Path,
    output_csv: Path,
    verbose: bool = False,
    level_idx: int = 2
) -> pd.DataFrame:
    """
    1) Load your roster (one row per batter for the upcoming season)
    2) Load the per-batter random‐effects summary
    3) Load the global effects JSON
    4) Assemble predicted means + intervals
    """
    import pandas as pd, json

    # ── 1) Read in
    roster = pd.read_csv(roster_csv)
    summary = pd.read_parquet(posterior_parquet)

    # ── 2) Load the globals
    globals_ = json.loads(global_effects_json.read_text())

    if verbose:
        print(f"[Debug] roster columns:  {roster.columns.tolist()}")
        print(f"[Debug] summary columns: {summary.columns.tolist()}")

    # ── 3) Merge in the random-effects
    #     (one row per batter_id in `summary`)
    df = roster.merge(
        summary,
        on="batter_id",
        how="left"
    )
    if verbose:
        print("[Debug] after merge df.columns:", df.columns.tolist())

    # ── 4) Compute contributions
    df["contrib_age"]   = globals_["beta_age"] * (
        df["age"] - globals_["median_age"]
    )
    df["contrib_level"] = globals_.get("beta_level", 0.0) * df["level_idx"]
    df["contrib_u"]     = df["u_q50"].fillna(0.0)

    # ── 5) Predicted mean
    df["pred_mean"] = (
        globals_["alpha"]
        + df["contrib_level"]
        + df["contrib_age"]
        + df["contrib_u"]
    )

    # ── 6) Confidence intervals (add the u‐quantiles)
    df["pred_lo95"] = df["pred_mean"] + df["u_q2.5"].fillna(0.0)
    df["pred_hi95"] = df["pred_mean"] + df["u_q97.5"].fillna(0.0)

    # ── 7) Write & return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df




if __name__ == '__main__':
    from src.utils.validation import (
        prediction_interval,
        bootstrap_prediction_interval,
    )
    from src.features.feature_engineering import feature_engineer

    BASE    = Path('data/models/saved_models')
    SUMMARY = BASE / 'posterior_summary.parquet'
    GLOBAL  = BASE / 'global_effects.json'
    ROSTER  = Path('data/Research Data Project/Research Data Project/exit_velo_validate_data.csv')
    RAW     = Path('data/Research Data Project/Research Data Project/exit_velo_project_data.csv')  # your training file
    OUT     = Path('data/predictions/exitvelo_predictions_2024.csv')

    # load raw data
    df_raw = pd.read_csv(RAW)
    df_fe    = feature_engineer(df_raw)
    print(df_fe.head())
    print(df_fe.columns)
    # load posterior summary
    df_post = pd.read_parquet(SUMMARY)
    # load global effects
    glob = json.loads(GLOBAL.read_text())
    

    # 1) Generate predictions + CI columns
    predict_df = predict_from_summaries(
        roster_csv=ROSTER,
        raw_csv=RAW, 
        posterior_parquet=SUMMARY,
        global_effects_json=GLOBAL,
        output_csv=OUT,
        verbose=True
    )
    print(predict_df.head())

    # 2) Empirical RMSE if 'exit_velo' present
    df_val = pd.read_csv(ROSTER)
    if 'exit_velo' in df_val.columns:
        y_true = df_val['exit_velo'].values
        y_pred = predict_df['pred_mean'].values
        rmse_val = np.sqrt(np.mean((y_pred - y_true)**2))
        print("\n--- empirical RMSE on validation set ---", rmse_val)

    # 3) Prepare a preds DataFrame for CI routines
    preds = predict_df['pred_mean'].to_frame(name='pred')
    lo95  = predict_df['pred_lo95'].values
    hi95  = predict_df['pred_hi95'].values


    # 4) Safely extract top hitters (requires a 'hitter_type' column)
    top_power, top_contact = get_top_hitters(predict_df, hitter_col="hitter_type", n=5, verbose=True)

    
