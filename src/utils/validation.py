"""
Generic K‑fold validator.

• Works for sklearn Pipelines *or* PyMC idata.
• Decides how to extract predictions based on
  the object returned by `fit_func`.
"""

import numpy as np
from sklearn.model_selection import KFold
import arviz as az

from __future__ import annotations
import pandas as pd
from sklearn.model_selection import KFold
from typing import Callable, List, Union


def _split_xy(df: pd.DataFrame):
    X = df.drop(columns=["exit_velo"])
    y = df["exit_velo"]
    return X, y


def _rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def run_kfold_cv(
    fit_func: Callable[[pd.DataFrame, pd.DataFrame], tuple],
    df: pd.DataFrame,
    k: int = 5,
    random_state: int = 0,
    **fit_kw
) -> List[float]:
    """
    fit_func(train_df, test_df, **fit_kw) -> (model_or_idata, rmse)
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    rmses: List[float] = []

    for train_idx, test_idx in kf.split(df):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        _, rmse = fit_func(train, test, **fit_kw)
        rmses.append(rmse)

    return rmses


# helper to score a *single* train/test split for idata
def rmse_pymc(idata: az.InferenceData, test_df: pd.DataFrame) -> float:
    """Posterior mean vs truth."""
    pred = (
        idata.posterior_predictive["y_obs"]
        .mean(("chain", "draw"))
        .values
    )
    return _rmse(pred, test_df["exit_velo"].values)

def run_kfold_cv(fit_func, df, k=5, random_state=0, **fit_kwargs):
    """
    Apply `fit_func(train_df, **fit_kwargs)` then evaluate on held-out.
    Returns list of held-out log_likelihoods or RMSEs.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in kf.split(df):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        idata = fit_func(train, **fit_kwargs)

        # posterior predictive on test
        ppc = az.from_pymc(posterior_predictive=idata, model=None)
        pred_mean = ppc.posterior_predictive["y_obs"].mean(("chain","draw")).values
        true = test["exit_velo"].values
        rmse = np.sqrt(((pred_mean - true)**2).mean())
        scores.append(rmse)
    return scores

def posterior_predictive_check(idata, df, batter_idx):
    """
    Plot observed vs. simulated exit-velo histograms.
    """
    import matplotlib.pyplot as plt
    obs = df["exit_velo"].values
    sim = idata.posterior_predictive["y_obs"].stack(samples=("chain","draw")).values.flatten()

    fig, ax = plt.subplots(1,2,figsize=(8,3))
    ax[0].hist(obs, bins=30); ax[0].set_title("Observed")
    ax[1].hist(sim, bins=30); ax[1].set_title("Simulated")
    fig.tight_layout()
    return fig





if __name__ == "__main__":
    from src.data.load_data import load_raw
    from src.features.preprocess import preprocess
    from src.models.hierarchical import fit_bayesian_hierarchical
    path = 'data/Research Data Project/Research Data Project/exit_velo_project_data.csv'
    df = load_raw(path)
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
    df_clean = preprocess(df)
    # show before and after
    print(f"Before: {df_train.head()}")
    print(f"After: {df_clean.head()}")

    idata = fit_bayesian_hierarchical(
    df_clean,
    batter_idx=df_clean.batter_id.cat.codes.values,
    level_idx=df_clean.level_idx.values,
    age_centered=df_clean.age_centered.values,
    mu_prior=90,      # use league ave +/–5 mph
    sigma_prior=5
    )

    posterior_predictive_check(idata, df_clean, df_clean.batter_id.cat.codes.values)


