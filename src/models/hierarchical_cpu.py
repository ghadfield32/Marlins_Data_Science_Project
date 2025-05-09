
import pymc as pm
import arviz as az
import numpy as np
from sklearn.compose import ColumnTransformer
from src.data.ColumnSchema import _ColumnSchema
from src.features.preprocess import transform_preprocessor

import time
from contextlib import contextmanager

# ── Context manager for timing ─────────────────────────────────────────
@contextmanager
def _timed_section(label: str):
    t0 = time.time()
    yield
    print(f"[{label}] finished in {time.time() - t0:,.1f} s")


def fit_bayesian_hierarchical(
    df_raw,
    transformer: ColumnTransformer,
    batter_idx: np.ndarray,
    level_idx: np.ndarray,
    season_idx: np.ndarray,
    pitcher_idx: np.ndarray,
    *,
    feature_list: list[str] | None = None,
    mu_mean: float  = 88.0,
    mu_sd:   float  = 30.0,
    sigma_prior: float = 10.0,
    draws: int      = 200,
    tune:  int      = 200,
    target_accept: float = 0.9,
    verbose: bool   = False,
    chains: int     = 1,
):
    """
    Hierarchical Bayesian model (CPU-only, no widgets).
    """

    # 1) Build design matrix & target
    cols = _ColumnSchema()
    if feature_list is None:
        feature_list = cols.model_features()

    X_all, y_ser = transform_preprocessor(df_raw, transformer)
    names = transformer.get_feature_names_out()
    X = X_all[:, np.isin(names, feature_list)]
    y = y_ser.values

    # 2) Hierarchical sizes
    n_bat  = int(batter_idx.max()) + 1
    n_lvl  = int(level_idx.max()) + 1
    n_seas = int(season_idx.max()) + 1
    n_pit  = int(pitcher_idx.max()) + 1
    n_feat = X.shape[1]
    if verbose:
        print(f"🔍 Data dims: {n_bat=} {n_lvl=} {n_seas=} {n_pit=} {n_feat=}")

    # 3) Model definition
    with pm.Model() as model:
        mu         = pm.Normal("mu", mu_mean, mu_sd)
        beta_level = pm.Normal("beta_level", 0, 5, shape=n_lvl)
        beta       = pm.Normal("beta",       0, 5, shape=n_feat)

        sigma_b    = pm.HalfNormal("sigma_b", sigma_prior)
        u_raw      = pm.Normal("u_raw", 0, 1, shape=n_bat)
        u          = pm.Deterministic("u", u_raw * sigma_b)

        sigma_seas = pm.HalfNormal("sigma_seas", sigma_prior)
        v_raw      = pm.Normal("v_raw", 0, 1, shape=n_seas)
        v          = pm.Deterministic("v", v_raw * sigma_seas)

        sigma_pit  = pm.HalfNormal("sigma_pit", sigma_prior)
        p_raw      = pm.Normal("p_raw", 0, 1, shape=n_pit)
        p          = pm.Deterministic("p", p_raw * sigma_pit)

        theta = (
            mu
            + beta_level[level_idx]
            + pm.math.dot(X, beta)
            + u[batter_idx]
            + v[season_idx]
            + p[pitcher_idx]
        )
        sigma_e = pm.HalfNormal("sigma_e", sigma_prior)
        pm.Normal("y_obs", theta, sigma_e, observed=y)

        # ── Sampling on CPU (plain text only) ────────────────────────────
        if verbose:
            print("▶ Starting MCMC sampling...")
        with _timed_section("compile+sample"):
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                # disable all progress bars/widgets:
                progressbar=False,
                idata_kwargs={"log_likelihood": ["y_obs"]},
            )
        if verbose:
            print("▶ Sampling complete.")

        # ── Posterior predictive sampling ────────────────────────────────
        if verbose:
            print("▶ Starting posterior predictive sampling...")
        with _timed_section("posterior_predictive"):
            idata.extend(
                pm.sample_posterior_predictive(idata, var_names=["y_obs"])
            )
        if verbose:
            print("▶ Posterior predictive complete.")

    # 4) Store feature names for downstream use
    idata.attrs["feature_names"] = [fn for fn in names if fn in feature_list]
    if verbose:
        samp = (
            idata.posterior_predictive["y_obs"]
                 .stack(s=("chain", "draw")).values
        )
        print("⚡ Posterior ppc (first 5):", samp[:5])

    return idata




# ───────────────────────────────────────────────────────────────────────
# 6. Smoke test (only run when module executed directly)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd, numpy as np, arviz as az
    from src.data.load_data import load_and_clean_data
    from src.features.feature_engineering import feature_engineer
    from src.features.preprocess import (
        fit_preprocessor,
        prepare_for_mixed_and_hierarchical,
    )
    from src.utils.hierarchical_utils import save_model, save_preprocessor
    from src.utils.posterior import posterior_to_frame
    from src.utils.bayesian_metrics import (
        compute_classical_metrics,
        compute_bayesian_metrics,
        compute_convergence_diagnostics,
        compute_calibration,
    )
    from src.utils.validation import (
        run_kfold_cv_pymc,
        rmse_pymc,
        posterior_predictive_check,
    )

    # Paths & data prep
    RAW      = Path("data/Research Data Project/Research Data Project/exit_velo_project_data.csv")
    OUT_NC   = Path("data/models/saved_models/exitvelo_hmc.nc")
    OUT_POST = Path("data/models/saved_models/posterior_summary.parquet")
    OUT_PRE  = Path("data/models/saved_models/preprocessor.joblib")

    df        = load_and_clean_data(RAW)
    df_fe     = feature_engineer(df)
    df_model  = prepare_for_mixed_and_hierarchical(df_fe)
    _, _, tf  = fit_preprocessor(df_model, model_type="linear", debug=False)

    # Index arrays
    b_idx = df_model["batter_id"].cat.codes.values
    l_idx = df_model["level_idx"].values
    s_idx = df_model["season_idx"].values
    p_idx = df_model["pitcher_idx"].values

    # Fit on CPU
    idata = fit_bayesian_hierarchical(
        df_model, tf, b_idx, l_idx, s_idx, p_idx,
        draws=10, tune=10, target_accept=0.95, chains=4, verbose=True
    )

    # Persist everything
    idata.attrs["median_age"] = df_model["age"].median()
    save_model(idata, OUT_NC)
    save_preprocessor(tf, OUT_PRE)
    posterior_to_frame(idata).to_parquet(OUT_POST)
    print("✅ training complete – artefacts written:")
    print("   •", OUT_NC)
    print("   •", OUT_POST)
    print("   •", OUT_PRE)
