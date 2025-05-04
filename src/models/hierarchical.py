
import jax
import pymc as pm
import arviz as az
import numpy as np

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

def fit_bayesian_hierarchical(
    df,
    batter_idx: np.ndarray,
    level_idx: np.ndarray,
    age_centered: np.ndarray,
    mu_prior: float,
    sigma_prior: float,
    *,
    use_ar1: bool = False,
    sampler: str = "nuts",          # "nuts", "nutpie", or "advi"
    draws: int = 1000,
    tune: int = 1000,
    advi_iters: int = 50_000,
):
    """
    Hierarchical Exit‑Velocity model with rich progress bars & ETAs.
    Returns a single ArviZ InferenceData object (posterior ⊕ PPC).
    """
    y      = df["exit_velo"].values
    n_bat  = df["batter_id"].nunique()
    n_lvl  = df["level_idx"].nunique()

    with pm.Model() as model:
        # ─── Priors ────────────────────────────────────────────────
        mu         = pm.Normal("mu", mu_prior, sigma_prior)
        beta_level = pm.Normal("beta_level", 0.0, 5.0, shape=n_lvl)
        beta_age   = pm.Normal("beta_age",   0.0, 1.0)
        sigma_b    = pm.HalfNormal("sigma_b", sigma_prior)
        sigma_e    = pm.HalfNormal("sigma_e", sigma_prior / 2)

        u = (
            pm.GaussianRandomWalk("u", sigma=sigma_b, shape=n_bat)
            if use_ar1 else
            pm.Normal("u", mu=0.0, sigma=sigma_b, shape=n_bat)
        )

        theta = (
            mu
            + beta_level[level_idx]
            + beta_age * age_centered
            + u[batter_idx]
        )
        pm.Normal("y_obs", mu=theta, sigma=sigma_e, observed=y)

        # ─── Inference ─────────────────────────────────────────────
        if sampler == "advi":
            print(f"Running ADVI ({advi_iters:,} iters) …")
            with _timed_section("ADVI"):
                approx = pm.fit(
                    n=advi_iters,
                    method="advi",
                    progressbar=True,
                )                    # PyMC shows a tqdm bar internally
            with _timed_section("Sampling from ADVI guide"):
                idata = approx.sample(draws=draws, progressbar=True)

        else:                        # HMC family
            total_steps = (draws + tune)
            print(f"Running {sampler.upper()} : {draws:,} draws + "
                  f"{tune:,} tune per chain (4 chains → {total_steps*4:,} steps)")
            sample_kwargs = dict(
                draws=draws, tune=tune, chains=4,
                target_accept=0.9, return_inferencedata=True,
                progressbar=True                      # PyMC uses tqdm
            )
            # select sampler backend
            if sampler == "nutpie":
                sample_kwargs["nuts_sampler"] = "nutpie"
            elif sampler == "jax":
                # Use JAX backend with GPU
                print("Using JAX on GPU for sampling")
                sample_kwargs["nuts_sampler"] = "numpyro"
                if jax.device_count("gpu") > 0:
                    print("GPU is available for sampling")
                else:
                    print("Warning: No GPU detected, using CPU instead")

            # Attempt JAX HMC, catch XLA dtype bug and fallback to nutpie
            with _timed_section("HMC sampling"):
                try:
                    idata = pm.sample(**sample_kwargs)
                except Exception as e:
                    # detect JAX XLARuntimeError by message or type
                    if "CpuCallback error" in str(e) or "Incorrect output dtype" in str(e):
                        print("⚠️  JAX sampler failed due to dtype bug; falling back to nutpie sampler")
                        sample_kwargs["nuts_sampler"] = "nutpie"
                        idata = pm.sample(**sample_kwargs)
                    else:
                        print(f"Error during sampling: {e}")
                        raise

        # ─── Posterior predictive ─────────────────────────────────
        n_ppc = y.size * draws
        print(f"Generating PPC ({draws:,} draws → {n_ppc:,} values) …")
        with _timed_section("Posterior predictive"):
            ppc = pm.sample_posterior_predictive(
                idata, var_names=["y_obs"], progressbar=True
            )

    return az.concat(idata, ppc)



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
    batter_idx = df_model["batter_id"].cat.codes.values
    level_idx = df_model["level_idx"].values
    age_centered = df_model["age_centered"].values

    # Fit the Bayesian hierarchical model
    idata = fit_bayesian_hierarchical(
        df_model, batter_idx, level_idx, age_centered,
        mu_prior=90, sigma_prior=5,
        sampler="jax",   #  <-- GPU NUTS
        draws=1000, tune=1000
    )

    print(idata)



