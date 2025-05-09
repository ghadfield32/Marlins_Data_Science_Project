"""
Train / compare four families on a 70‑30 split.

Run:
    python -m src.train
"""
from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load_data import load_and_clean_data
from src.features.preprocess import preprocess

from src.models.linear import fit_ridge
from src.models.gbm   import fit_gbm
from src.models.mixed import fit_mixed
from src.models.hierarchical import fit_bayesian_hierarchical

RAW_PATH = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"


def main():
    df_raw   = load_and_clean_data(RAW_PATH)
    df_clean = preprocess(df_raw)

    train_df, test_df = train_test_split(
        df_clean, test_size=0.30, random_state=42, stratify=df_clean["level_abbr"]
    )

    # ––– A  Ridge  –––––––––––––––––––––––––––––––
    _, rmse_ridge = fit_ridge(train_df, test_df)
    print(f"Ridge RMSE ……  {rmse_ridge:5.2f} mph")

    # ––– B  Gradient‑Boost  ––––––––––––––––––––––
    _, rmse_gbm = fit_gbm(train_df, test_df)
    print(f"XGBoost RMSE … {rmse_gbm:5.2f} mph")

    # ––– C  Mixed‑Effects  –––––––––––––––––––––––
    _, rmse_mixed = fit_mixed(train_df, test_df)
    print(f"Mixed‑LM RMSE  {rmse_mixed:5.2f} mph")

    # ––– D  Bayesian Hierarchical (quick sample) –
    idata = fit_bayesian_hierarchical(
        train_df,
        batter_idx=train_df.batter_id.astype("category").cat.codes.values,
        level_idx=train_df.level_idx.values,
        age_centered=train_df.age_centered.values,
        mu_prior=90,
        sigma_prior=5,
        draws=500, tune=500   # short run for demo
    )
    from src.utils.validation import rmse_pymc
    rmse_bayes = rmse_pymc(idata, test_df)
    print(f"PyMC RMSE ……  {rmse_bayes:5.2f} mph")


if __name__ == "__main__":
    main()

