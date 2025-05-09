

import json
from pathlib import Path

import arviz as az
import pandas as pd
import numpy as np

from src.utils.posterior import align_batter_codes

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
    posterior_parquet: Path,
    global_effects_json: Path,
    output_csv: Path,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load saved model summaries and raw roster, merge random effects,
    compute point predictions and intervals for validation set,
    write out predictions CSV, and return the DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['season','batter_id','age','level_idx','age_centered','batter_idx',
         'u_q50','u_q2.5','u_q97.5',
         'contrib_age','contrib_level','contrib_u','contrib_features',
         'pred_mean','pred_lo95','pred_hi95']
    """
    # 1) Load posterior random effects (batter-level)
    df_post = pd.read_parquet(posterior_parquet)
    if verbose:
        print(f"[Load] posterior_summary: {df_post.shape} rows")

    # 2) Load validation roster
    df_roster = pd.read_csv(roster_csv)
    if verbose:
        print(f"[Load] validation roster: {df_roster.shape} rows, columns: {df_roster.columns.tolist()}")

    # 3) Load global effects
    glob        = json.loads(global_effects_json.read_text())
    post_mu     = glob['mu_mean']
    beta_age    = glob['beta_age']
    beta_level  = glob['beta_level'][2]  # MLB-level coefficient
    med_age     = glob['median_age']
    if verbose:
        print(f"[Global Effects] mu={post_mu}, beta_age={beta_age}, "
              f"beta_level={beta_level}, median_age={med_age}")

    # 4) Prepare minimal validation features
    df_val = simplified_prepare_validation(df_roster, med_age, verbose)

    # 5) Align batter codes and merge random effects
    df_val['batter_idx'] = align_batter_codes(df_val, df_post['batter_idx'])
    df = df_val.merge(df_post, on='batter_idx', how='left')
    if verbose:
        print(f"[Merge] merged validation with posterior: {df.shape}")

    # 6) Compute contributions & predictions
    # 6a) Age contribution
    df['contrib_age']   = beta_age * df['age_centered']
    # 6b) Level (MLB) contribution
    df['contrib_level'] = beta_level
    # 6c) Batter random effect (median)
    df['contrib_u']     = df['u_q50']

    # 6d) Point‐estimate
    df['pred_mean']  = post_mu + df['contrib_age'] + df['contrib_level'] + df['contrib_u']

    # 6e) 95% credible‐interval bounds
    df['pred_lo95']  = post_mu + df['contrib_age'] + df['contrib_level'] + df['u_q2.5']
    df['pred_hi95']  = post_mu + df['contrib_age'] + df['contrib_level'] + df['u_q97.5']

    # 6f) Placeholder for any other feature contributions
    df['contrib_features'] = 0.0

    # 7) Persist predictions CSV
    df.to_csv(output_csv, index=False)
    if verbose:
        print(f"[Save] Predictions written to {output_csv}")

    # 8) Return for downstream inspection or metrics
    return df




if __name__ == '__main__':
    from src.utils.validation import (
        prediction_interval,
        bootstrap_prediction_interval,
    )

    BASE    = Path('data/models/saved_models')
    SUMMARY = BASE / 'posterior_summary.parquet'
    GLOBAL  = BASE / 'global_effects.json'
    ROSTER  = Path('data/Research Data Project/Research Data Project/exit_velo_validate_data.csv')
    OUT     = Path('data/predictions/exitvelo_predictions_2024.csv')

    # 1) Generate predictions + CI columns
    predict_df = predict_from_summaries(
        roster_csv=ROSTER,
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

    print("\nTop Power Hitters (if any):\n", top_power)
    print("\nTop Contact Hitters (if any):\n", top_contact)

