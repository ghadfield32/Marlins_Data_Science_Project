
import pymc as pm
import arviz as az
import numpy as np
from src.data.ColumnSchema import _ColumnSchema
from src.features.preprocess import transform_preprocessor
# ── Attempt to import JAX ──────────────────────────────────────
USE_JAX = True
try:
    import jax
    # Debug: confirm what module is loaded
    print(f"🔍 JAX module: {jax!r}")
    print(f"🔍 JAX path:   {getattr(jax, '__file__', 'builtin')}")
    # Ensure version attribute exists (guards circular-import)
    if not hasattr(jax, "__version__"):
        raise ImportError("jax.__version__ missing—possible circular import")
    print(f"✅ JAX version: {jax.__version__}")
    # Enable 64-bit floats on GPU/CPU
    jax.config.update("jax_enable_x64", True)
except Exception as e:
    USE_JAX = False
    print(f"⚠️  Warning: could not import JAX ({e}). Falling back to CPU sampling.")

import pymc as pm
import arviz as az
import numpy as np
import pandas as pd


# Configure JAX for GPU use and X64 precision
jax.config.update("jax_enable_x64", True)
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("GPU count:", jax.device_count("gpu"))
print("Default backend:", jax.default_backend())

import logging, pymc.sampling.jax as pmjax


# ── NEW: fit_bayesian_hierarchical with timing & ETAs ────────────────
import time
from contextlib import contextmanager
from tqdm.auto import tqdm          # auto‑selects rich bar in Jupyter / CLI

@contextmanager
def _timed_section(label: str):
    """Context manager that prints elapsed time for a code block."""
    t0 = time.time()
    yield
    dt = time.time() - t0
    print(f"[{label}] finished in {dt:,.1f} s")
    



# src/models/hierarchical.py

def fit_bayesian_hierarchical(
    df_raw,
    transformer,
    batter_idx: np.ndarray,
    level_idx: np.ndarray,
    *,
    feature_list: list[str] | None = None,
    use_ar1: bool   = False,
    mu_prior: float = 0.0,
    sigma_prior: float = 1.0,
    sampler: str    = "nuts",
    draws: int      = 1000,        # bumped from 5 → 1000
    tune: int       = 1000,        # bumped from 5 → 1000
    target_accept: float = 0.95,   # raised from 0.9 → 0.95
    rhat_threshold: float = 1.01,
    ess_threshold: int     = 100,
):
    """
    Hierarchical EV model with:
      • Non-centered random effects for u
      • Longer sampling (draws/tune=1000)
      • Higher target_accept to reduce divergences
    """
    cols = _ColumnSchema()
    if feature_list is None:
        feature_list = cols.model_features()

    # 1) transform inputs
    X_all, y_ser = transform_preprocessor(df_raw, transformer)
    names        = transformer.get_feature_names_out()
    mask = np.array([n in feature_list for n in names])
    X    = X_all[:, mask]
    y    = y_ser.values

    # 2) dims
    n_obs, n_feat = X.shape
    n_bat         = int(batter_idx.max() + 1)
    n_lvl         = int(level_idx.max()   + 1)

    # 3) build model
    with pm.Model() as model:
        mu         = pm.Normal("mu", mu_prior, sigma_prior)
        beta_level = pm.Normal("beta_level", 0, 1, shape=n_lvl)
        beta       = pm.Normal("beta", 0, 1, shape=n_feat)
        sigma_b    = pm.HalfNormal("sigma_b", sigma_prior)

        # --- NON-CENTERED u: robust to funnel geometry ---
        u_raw = pm.Normal("u_raw", 0, 1, shape=n_bat)
        u     = pm.Deterministic("u", u_raw * sigma_b)

        theta = mu + beta_level[level_idx] + pm.math.dot(X, beta) + u[batter_idx]
        sigma_e = pm.HalfNormal("sigma_e", sigma_prior)
        pm.Normal("y_obs", mu=theta, sigma=sigma_e, observed=y)

        # --- Sampling with stronger defaults ---
        with _timed_section("Sampling"):
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=4,
                target_accept=target_accept,
                nuts_sampler="numpyro" if (sampler=="jax" and USE_JAX) else "nuts",
                progressbar=True,
            )
        with _timed_section("Posterior Predictive"):
            ppc = pm.sample_posterior_predictive(idata, var_names=["y_obs"])

    # attach ppc
    idata.extend(ppc)

    # 4) diagnostics
    summary = az.summary(idata,
                         var_names=["mu", "beta", "beta_level", "sigma_b", "sigma_e"],
                         round_to=2)
    bad_rhat = summary[summary["r_hat"] > rhat_threshold]
    bad_ess  = summary[summary["ess_bulk"] < ess_threshold]
    if not bad_rhat.empty:
        print(f"⚠️  {len(bad_rhat)} parameters with R̂ > {rhat_threshold}")
    if not bad_ess.empty:
        print(f"⚠️  {len(bad_ess)} parameters with ESS < {ess_threshold}")
    if bad_rhat.empty and bad_ess.empty:
        print("✅  All sampler diagnostics OK")

    return idata



# ───────────────────────────────────────────────────────────────────────
# 6. Smoke test (only run when module executed directly)
# data:
#     Data Used in the Hierarchical Model
# Exit Velocity Measurements

# Your outcome variable y is each batted-ball’s exit velocity, as recorded by Statcast—i.e., the speed (mph) at which the ball leaves the bat
# baseballsavant.com
# . Statcast began tracking exit velocity league-wide in 2015, using high-speed cameras and radar to measure every play
# Wikipedia
# .
# Covariates: Age and Competition Level

# You include each player’s centered age (age_centered) to capture how batting strength changes with age. Centering (subtracting the median) improves model convergence and interpretability. The discrete competition levels (level_idx: AA=0, AAA=1, MLB=2) let you estimate systematic differences in exit velocity across minor-league versus major-league play.
# Random Effects: Batter Identity

# By treating batter_id as a categorical random effect (u[batter_idx]), you allow each hitter to have his own baseline deviation from the global mean. This “partial pooling” borrows strength across batters—shrinking estimates for low-sample hitters toward the overall mean—so rarer batters aren’t grossly over- or under-estimated
# PyMC
# .
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_raw
    from src.features.feature_engineering import feature_engineer
    from src.features.preprocess import fit_preprocessor, prepare_for_mixed_and_hierarchical
    from src.models.hierarchical_utils import save_model, load_model

    MODEL_PATH = "data/models/saved_models/exitvelo_hmc.nc"

    # 1) Load raw and featurize
    raw_path = Path("data/Research Data Project/Research Data Project/exit_velo_project_data.csv")
    df = load_raw(raw_path)
    df_fe = feature_engineer(df)

    # 2) Prepare for mixed/hierarchical (drops bunts, NA, sets batter_id category)
    df_model = prepare_for_mixed_and_hierarchical(df_fe)

    # 3) Fit the same preprocessor we’ll use downstream
    #    – this gives us X_mat (unused here), y_train (also unused),
    #      and the ColumnTransformer we must pass to our PyMC model.
    _, _, transformer = fit_preprocessor(
        df_model,
        model_type="linear",  # or whichever you prefer
        debug=True            # helpful to see its internals
    )

    # 4) Extract the index arrays for random‐effects
    batter_idx = df_model["batter_id"].cat.codes.values
    level_idx  = df_model["level_idx"].values

    # 5) Quick smoke-run of the Bayes hierarchy
    idata = fit_bayesian_hierarchical(
        df_model,
        transformer,
        batter_idx,
        level_idx,
        sampler="jax",   # use GPU‐accelerated Nuts
        draws=5,
        tune=5,
    )
    print(idata)

    # 6) Save & reload roundtrip
    save_model(idata, MODEL_PATH)
    _ = load_model(MODEL_PATH)
