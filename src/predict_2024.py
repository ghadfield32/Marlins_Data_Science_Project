import pandas as pd, numpy as np, arviz as az
from pathlib import Path
from src.utils.posterior import align_batter_codes

POSTERIOR = Path("data/models/saved_models/posterior_summary.parquet")
ROSTER    = Path("data/Research Data Project/Research Data Project/exit_velo_validate_data.csv")
OUTPUT    = Path("data/predictions/exitvelo_predictions_2024.csv")

# ---------- 1 · load artefacts ----------
df_post   = pd.read_parquet(POSTERIOR)               # tidy posterior summary
df_roster = pd.read_csv(ROSTER)                      # season, batter_id, age

# median age from training (hard‑code or store separately)
MEDIAN_AGE = 26.0

# ---------- 2 · static effects ----------
post_mu   = df_post["u_mean"].mean()   # global μ̂ (strictly you’d read from idata)
beta_age  =  df_post.attrs.get("beta_age", -0.05)  # store during training
beta_lvl  = 0.0          # MLB reference level → 0 by identifiability choice

# ---------- 3 · merge random effects ----------
codes = align_batter_codes(df_roster, df_post["batter_idx"])
df_roster["batter_idx"] = codes

df_merged = df_roster.merge(df_post, on="batter_idx", how="left")

# unseen batters → μ̂ with wider σ (assume N(0, σ_b²))
global_sigma_b = df_post["u_sd"].mean()
df_merged["u_mean"].fillna(0.0, inplace=True)      # shrink to zero
df_merged["u_sd"].fillna(global_sigma_b, inplace=True)

# ---------- 4 · point & interval predictions ----------
df_merged["age_centered"] = df_merged["age"] - MEDIAN_AGE
df_merged["pred_mean"] = (post_mu
                          + beta_lvl
                          + beta_age * df_merged["age_centered"]
                          + df_merged["u_mean"])

z95 = 1.96
df_merged["pred_lo95"] = df_merged["pred_mean"] - z95 * df_merged["u_sd"]
df_merged["pred_hi95"] = df_merged["pred_mean"] + z95 * df_merged["u_sd"]

# ---------- 5 · export ----------
cols_out = ["season", "batter_id", "pred_mean", "pred_lo95", "pred_hi95"]
df_merged[cols_out].to_csv(OUTPUT, index=False)
print("📄 Predictions written →", OUTPUT)
