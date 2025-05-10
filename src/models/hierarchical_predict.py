

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
    *,
    roster_csv: Path,
    raw_csv: Path,
    posterior_parquet: Path,
    global_effects_json: Path,
    output_csv: Path,
    verbose: bool = False,
    level_idx: int = 2,  # allow override if you ever want a different level
) -> pd.DataFrame:
    # --- 1) Load artefacts ------------------------------------------------
    df_post   = pd.read_parquet(posterior_parquet)
    df_roster = pd.read_csv(roster_csv)
    df_raw    = pd.read_csv(raw_csv)

    # --- 2) Load & validate global effects -------------------------------
    try:
        glob = json.loads(global_effects_json.read_text())
    except FileNotFoundError:
        raise ValueError(f"Global-effects file {global_effects_json} not found")

    if not glob:
        raise ValueError(
            f"{global_effects_json} is empty – missing Intercept/beta_age/beta_level"
        )

    # --- 2a) Inspect beta_level representation (for debugging) -----------
    raw_bl = glob.get("beta_level", None)
    if verbose:
        print(f"[Debug] loaded beta_level of type {type(raw_bl).__name__}: {raw_bl}")

    # --- 2b) Robust retrieval of beta_level -----------------------------
    if isinstance(raw_bl, dict):
        # dict may use string keys or possibly int keys
        beta_level = raw_bl.get(str(level_idx),
                        raw_bl.get(level_idx, 0.0))
    elif isinstance(raw_bl, (list, tuple)):
        # list index
        try:
            beta_level = raw_bl[level_idx]
        except (IndexError, TypeError):
            beta_level = 0.0
    else:
        # unrecognized format
        beta_level = 0.0

    # pull the rest
    post_mu  = glob.get("mu_mean",    0.0)
    beta_age = glob.get("beta_age",    0.0)
    med_age  = glob.get(
        "median_age",
        float(df_raw.get('age', pd.Series()).median() or 0.0)
    )

    # --- 3) Feature engineering (unchanged) ------------------------------
    df_fe = feature_engineer(df_raw)
    mapping = (
        df_fe[df_fe['season']==2023][['batter_id','hitter_type']]
             .dropna()
             .groupby('batter_id')['hitter_type']
             .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
             .to_dict()
    )
    df_roster['hitter_type'] = df_roster['batter_id'].map(mapping).fillna("UNKNOWN")

    # --- 4) Validation prep & merge --------------------------------------
    df_val = simplified_prepare_validation(df_roster, med_age, verbose)
    df_val['batter_idx'] = align_batter_codes(df_val, df_post['batter_idx'])
    df = df_val.merge(df_post, on='batter_idx', how='left')

    # --- 5) Compute contributions & predictions --------------------------
    df['contrib_age']   = beta_age   * df['age_centered']
    df['contrib_level'] = beta_level
    df['contrib_u']     = df['u_q50']

    df['pred_mean'] = post_mu + (
        df['contrib_age'] + df['contrib_level'] + df['contrib_u']
    )
    df['pred_lo95'] = post_mu + df['contrib_age'] + df['contrib_level'] + df['u_q2.5']
    df['pred_hi95'] = post_mu + df['contrib_age'] + df['contrib_level'] + df['u_q97.5']

    # --- 6) Persist with auto-mkdir + locked column order ---------------
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "season","batter_id","age","hitter_type","level_idx","age_centered",
        "batter_idx","u_q2.5","u_q50","u_q97.5",
        "contrib_age","contrib_level","contrib_u",
        "pred_mean","pred_lo95","pred_hi95"
    ]
    df[cols].to_csv(output_csv, index=False)
    if verbose:
        print(f"[Save] predictions → {output_csv}")

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

    
