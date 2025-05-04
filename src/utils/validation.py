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
import statsmodels as sm

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




def prediction_interval(model, X, alpha=0.05, method='linear'):
    """
    Compute prediction intervals for a model.
    """
    if method == 'linear':
        # For OLS and Ridge
        X_const = sm.add_constant(X)
        preds = model.get_prediction(X_const)
        pred_int = preds.conf_int(alpha=alpha)
        return preds.predicted_mean, pred_int[:, 0], pred_int[:, 1]
    elif method == 'bayesian':
        # For Bayesian models
        hdi = az.hdi(model, hdi=1 - alpha)
        return (
            hdi.posterior_predictive.y_obs.sel(hdi=f"{alpha/2*100}%"),
            hdi.posterior_predictive.y_obs.sel(hdi=f"{(1-alpha/2)*100}%")
        )
    elif method == 'gbm':
        # For XGBoost quantile regression
        lower = model.predict(X, pred_contribs=False, iteration_range=(0, model.best_iteration))
        upper = model.predict(X, pred_contribs=False, iteration_range=(0, model.best_iteration))
        return lower, upper  # Replace with actual quantile regression
    else:
        raise ValueError("Method not supported")

# Example for bootstrapping GBM
def bootstrap_prediction_interval(model, X, n_bootstraps=1000, alpha=0.05):
    preds = np.zeros((n_bootstraps, X.shape[0]))
    for i in range(n_bootstraps):
        indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        preds[i] = model.predict(X[indices])
    lower = np.percentile(preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
    return lower, upper


# ───────────────────────────────────────────────────────────────────────
# 6. Smoke test (only run when module executed directly)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.load_data import load_raw
    from src.features.feature_engineering import feature_engineer
    from src.features.preprocess import prepare_for_mixed_and_hierarchical

    raw_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
    df = load_raw(raw_path)
    df_fe = feature_engineer(df)

    # Prepare the DataFrame
    df_model = prepare_for_mixed_and_hierarchical(df_fe)

    # Extract arrays for PyMC
    batter_idx   = df_model["batter_id"].cat.codes.values
    level_idx    = df_model["level_idx"].values
    age_centered = df_model["age_centered"].values

    # Fit the Bayesian hierarchical model
    idata = fit_bayesian_hierarchical(
        df_model, batter_idx, level_idx, age_centered,
        mu_prior=90, sigma_prior=5,
        sampler="jax",   #  <-- GPU NUTS
        draws=1000, tune=1000
    )

    print(idata)

    posterior_predictive_check(idata, df_model, df_model.batter_id.cat.codes.values)

    # For Bayesian model:
    lower, upper = prediction_interval(idata, test_df, method='bayesian')
    print(f"Bayesian 95% Prediction Interval: {lower.mean():.2f}–{upper.mean():.2f} mph")

    # For Ridge model:
    pred, lower, upper = prediction_interval(model_ridge, X_test, method='linear')
    print(f"Ridge 95% Prediction Interval: {lower[0]:.2f}–{upper[0]:.2f} mph")
