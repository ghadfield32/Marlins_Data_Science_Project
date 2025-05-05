from pathlib import Path
import pandas as pd, numpy as np, arviz as az
from src.data.load_data import load_raw
from src.features.feature_engineering import feature_engineer
from src.features.preprocess import (fit_preprocessor,
                                     prepare_for_mixed_and_hierarchical)
from src.models.hierarchical import fit_bayesian_hierarchical
from src.models.hierarchical_utils import save_model
from src.utils.posterior import posterior_to_frame

RAW   = Path("data/Research Data Project/Research Data Project/exit_velo_project_data.csv")
OUT_NC = Path("data/models/saved_models/exitvelo_hmc.nc")
OUT_POST = Path("data/models/saved_models/posterior_summary.parquet")

# 1 · prep
df = load_raw(RAW)
df_fe = feature_engineer(df)
df_model = prepare_for_mixed_and_hierarchical(df_fe)

_, _, tf = fit_preprocessor(df_model, model_type="linear", debug=False)

b_idx = df_model["batter_id"].cat.codes.values
l_idx = df_model["level_idx"].values

# 2 · fit
idata = fit_bayesian_hierarchical(df_model, tf, b_idx, l_idx,
                                  sampler="jax", draws=1000, tune=1000)

# 3 · persist
save_model(idata, OUT_NC)
posterior_to_frame(idata).to_parquet(OUT_POST)
print("✅ training complete – artefacts written:")
print("   •", OUT_NC)
print("   •", OUT_POST)
