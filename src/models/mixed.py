
"""
Frequentist mixed‑effects model using statsmodels MixedLM.

Formula implemented:
    exit_velo ~ 1 + level_ord + age_centered
              + (1 | batter_id)

We rely on columns already produced by preprocess():
    • level_idx  (0,1,2)   – ordinal
    • age_centered
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_mixed(train: pd.DataFrame,
              test: pd.DataFrame):
    """Return (fitted model, RMSE on test)."""
    # statsmodels wants a *single* DataFrame with all cols
    # so we concatenates and keep row positions for slicing
    combined = pd.concat([train, test], axis=0)
    # ensure categorical dtype
    combined["level_ord"] = combined["level_idx"].astype(int)

    mdl = smf.mixedlm(
        formula="exit_velo ~ 1 + level_ord + age_centered",
        data=combined.iloc[: len(train)],
        groups=combined.iloc[: len(train)]["batter_id"]
    ).fit(reml=False)

    # predict on test set
    pred = mdl.predict(exog=combined.iloc[len(train):])
    true = test["exit_velo"].values
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    return mdl, rmse

if __name__ == "__main__":
    from src.data.load_data import load_raw
    from src.features.feature_engineering import feature_engineer
    from src.features.preprocess import prepare_for_mixed_and_hierarchical
    from sklearn.model_selection import train_test_split

    raw_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
    df = load_raw(raw_path)
    df_fe = feature_engineer(df)

    # Prepare and split
    df_model = prepare_for_mixed_and_hierarchical(df_fe)
    train_df, test_df = train_test_split(df_model, test_size=0.2, random_state=42)
 
    # Fit mixed-effects
    mixed_model, rmse_mixed = fit_mixed(train_df, test_df)
    print(f"Mixed-effects model RMSE: {rmse_mixed:.4f}")

    # In the smoke test section: P-Value Checks for Mixed-Effects Models
    print("Mixed-effects model summary:\n", mixed_model.summary())
