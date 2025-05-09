"""
Generic K‑fold validator.

• Works for sklearn Pipelines *or* PyMC idata.
• Decides how to extract predictions based on
  the object returned by `fit_func`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from arviz import InferenceData
import arviz as az
from typing import Callable, List, Union
import statsmodels as sm



def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.mean((a - b) ** 2))


def run_kfold_cv(
    fit_func: Callable[[pd.DataFrame, pd.DataFrame], tuple],
    df: pd.DataFrame,
    k: int = 5,
    random_state: int = 0,
    **fit_kw
) -> List[float]:
    """
    Sklearn-style K-fold CV:
      fit_func(train_df, test_df, **fit_kw) -> (model, rmse)
    Returns a list of rmse scores.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    rmses: List[float] = []
    for train_idx, test_idx in kf.split(df):
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        _, rmse = fit_func(train, test, **fit_kw)
        rmses.append(rmse)
    return rmses



def run_kfold_cv_pymc(
    fit_func: Callable[[pd.DataFrame], InferenceData],
    df: pd.DataFrame,
    k: int = 5,
    random_state: int = 0
) -> list[float]:
    """
    PyMC K-fold CV: 
      fit_func(train_df) -> ArviZ InferenceData with posterior a[group] & sigma.
    Returns RMSE on each hold-out fold.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    rmses: list[float] = []

    for train_idx, test_idx in kf.split(df):
        train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
        idata = fit_func(train_df)

        # 1) grab posterior samples of the group intercepts and noise σ
        #    stack chain+draw into one dim of length M
        a_samples = (
            idata.posterior["a"]
            .stack(sample=("chain", "draw"))    # dims now ("group","sample")
            .transpose("sample", "group")       # shape (M, n_groups)
            .values
        )
        sigma_samples = (
            idata.posterior["sigma"]
            .stack(sample=("chain", "draw"))    # shape (M,)
            .values
        )

        # 2) for each test row, pull the intercept for its group
        groups = test_df["group"].values       # shape (n_test,)
        # a_samples[:, groups] → shape (M, n_test)
        pred_samples = a_samples[:, groups]

        # 3) point estimate = posterior mean for each test row
        pred_mean = pred_samples.mean(axis=0)  # shape (n_test,)

        # 4) compute RMSE against true exit_velo
        true_vals = test_df["exit_velo"].values
        rmse = np.sqrt(np.mean((pred_mean - true_vals) ** 2))
        rmses.append(rmse)

    return rmses




# helper to score a *single* train/test split for idata
def rmse_pymc(idata: InferenceData, test_df: pd.DataFrame) -> float:
    """
    Compute RMSE by comparing posterior-mean predictions
    (using the group intercept `a`) against observed `exit_velo`.
    """
    # 1) Stack (chain, draw) into a single "sample" dimension
    a_samples = (
        idata.posterior["a"]
        .stack(sample=("chain", "draw"))    # dims → ("group","sample")
        .transpose("sample", "group")       # shape → (M, n_groups)
        .values
    )

    # 2) Gather the sampled intercepts for each test row
    groups = test_df["group"].values       # shape → (n_test,)
    pred_samples = a_samples[:, groups]    # shape → (M, n_test)

    # 3) Posterior mean prediction for each test instance
    pred_mean = pred_samples.mean(axis=0)  # shape → (n_test,)

    # 4) Compute and return RMSE versus the true values
    true_vals = test_df["exit_velo"].values
    return np.sqrt(np.mean((pred_mean - true_vals) ** 2))


def posterior_predictive_check(idata, df, batter_idx):
    """
    Plot observed vs. simulated exit-velo histograms.
    """
    import matplotlib.pyplot as plt
    obs = df["exit_velo"].values
    sim = (
        idata.posterior_predictive["y_obs"]
        .stack(samples=("chain", "draw"))
        .values.flatten()
    )

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].hist(obs, bins=30)
    ax[0].set_title("Observed")
    ax[1].hist(sim, bins=30)
    ax[1].set_title("Simulated")
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
    """
    Compute nonparametric bootstrap confidence intervals for predictions.

    Must support either:
      - model.predict(X) -> array_like shape (n_obs,)
      - model.get_prediction(X).predicted_mean -> array_like shape (n_obs,)

    Returns lower, upper arrays of shape (n_obs,) at the given alpha level.
    """
    import numpy as np

    # Determine number of observations
    try:
        n_obs = X.shape[0]
    except Exception:
        raise ValueError("X must have a valid `.shape[0]`")

    # Allocate storage
    all_preds = np.zeros((n_bootstraps, n_obs))

    for i in range(n_bootstraps):
        # Sample indices with replacement
        idx = np.random.randint(0, n_obs, size=n_obs)
        sampled = X.iloc[idx] if hasattr(X, "iloc") else X[idx]

        # Dispatch to correct predict method
        if hasattr(model, "predict"):
            preds_i = model.predict(sampled)
        elif hasattr(model, "get_prediction"):
            result = model.get_prediction(sampled)
            preds_i = getattr(result, "predicted_mean", None)
            if preds_i is None:
                raise AttributeError(
                    "get_prediction(...) did not return `predicted_mean`"
                )
        else:
            raise AttributeError(
                f"Model {model!r} has no `predict` or `get_prediction` method"
            )

        all_preds[i, :] = preds_i

    # Compute CI percentiles
    lower = np.percentile(all_preds, 100 * (alpha / 2), axis=0)
    upper = np.percentile(all_preds, 100 * (1 - alpha / 2), axis=0)
    return lower, upper






if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import arviz as az
    import statsmodels.api as sm
    import pymc as pm
    from sklearn.linear_model import LinearRegression

    # from src.utils.validation import (
    # run_kfold_cv_pymc, prediction_interval, 
    # bootstrap_prediction_interval, rmse_pymc, 
    # posterior_predictive_check
    # )

    # 1) Synthetic data for sklearn
    np.random.seed(42)
    X = np.linspace(0, 10, 40)
    y = 2 * X + 1 + np.random.normal(0, 1, size=X.shape)
    df_skl = pd.DataFrame({"feature": X, "exit_velo": y})

    def fit_sklearn(train, test):
        X_tr, y_tr = train[["feature"]], train["exit_velo"]
        X_te, y_te = test[["feature"]], test["exit_velo"]
        m = LinearRegression().fit(X_tr, y_tr)
        preds = m.predict(X_te)
        rmse = np.sqrt(np.mean((preds - y_te) ** 2))
        return m, rmse

    print("\n--- sklearn CV ---")
    print("RMSEs:", run_kfold_cv(fit_sklearn, df_skl, k=4))

    # 2) Synthetic hierarchical data for PyMC
    #    (3 groups, group‐level intercepts)
    n_per = 20
    groups = np.repeat(np.arange(3), n_per)
    true_alpha = np.array([1.0, 3.0, 5.0])
    y_hier = true_alpha[groups] + np.random.normal(0, 1, size=groups.size)
    df_hier = pd.DataFrame({"exit_velo": y_hier, "group": groups})

    def fit_hierarchical(train_df: pd.DataFrame) -> az.InferenceData:
        """
        Fit a hierarchical model and append posterior_predictive samples for y_obs.
        Returns:
            InferenceData with .posterior and .posterior_predictive["y_obs"].
        """
        coords = {"group": np.unique(train_df["group"])}
        with pm.Model(coords=coords) as model:
            mu_a = pm.Normal("mu_a", 0, 5)
            sigma_a = pm.HalfNormal("sigma_a", 5)
            a = pm.Normal("a", mu=mu_a, sigma=sigma_a, dims="group")
            sigma = pm.HalfNormal("sigma", 1)
            y_grp = a[train_df["group"].values]
            pm.Normal("y_obs", mu=y_grp, sigma=sigma,
                    observed=train_df["exit_velo"].values)

            # 1) Draw posterior samples
            idata = pm.sample(
                draws=500, tune=500,
                chains=2, cores=1,
                return_inferencedata=True,
                progressbar=False
            )

            # 2) Generate posterior_predictive samples for 'y_obs'
            pm.sample_posterior_predictive(
                idata,
                model=model,
                var_names=["y_obs"],
                extend_inferencedata=True,
                random_seed=42,
                progressbar=False
            )

        # 3) Return the enriched InferenceData
        return idata


    print("\n--- PyMC hierarchical CV ---")
    bayes_rmse = run_kfold_cv_pymc(fit_hierarchical, df_hier, k=3)
    print("Bayesian RMSEs:", bayes_rmse)

    # 3) Test prediction_interval (linear OLS)
    df_ols = df_skl.copy()
    df_ols["const"] = 1.0
    ols = sm.OLS(df_ols["exit_velo"], df_ols[["const", "feature"]]).fit()
    mean, lo, hi = prediction_interval(ols, df_skl[["feature"]], method="linear")
    print("\n--- OLS prediction_interval shapes ---")
    print("mean:", mean.shape, "lower:", lo.shape, "upper:", hi.shape)

    # 4) Test bootstrap_prediction_interval with LinearRegression
    lr = LinearRegression().fit(df_skl[["feature"]], df_skl["exit_velo"])
    lower, upper = bootstrap_prediction_interval(lr, df_skl[["feature"]], n_bootstraps=200)
    print("\n--- bootstrap_prediction_interval shapes ---")
    print("lower:", lower.shape, "upper:", upper.shape)

    # 5) rmse_pymc + posterior_predictive_check on a single InferenceData
    #    (we can re-use the last fold of the hierarchical example)
    last_idata = fit_hierarchical(df_hier)
    rmse_val = rmse_pymc(last_idata, df_hier)
    print("\n--- rmse_pymc on full data ---", rmse_val)

    # 6) posterior_predictive_check plot
    print("\n--- posterior_predictive_check (figure) ---")
    fig = posterior_predictive_check(last_idata, df_hier, batter_idx=None)
    fig.savefig("ppc_hist.png")
    print("Saved histogram to ppc_hist.png")
