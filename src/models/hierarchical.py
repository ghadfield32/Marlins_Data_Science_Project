
import pymc as pm
import arviz as az
import numpy as np
import json  # Add json import for allocation result
from sklearn.compose import ColumnTransformer
from src.data.ColumnSchema import _ColumnSchema
from src.features.preprocess import transform_preprocessor

# ─── Apply our JAX GPU-memory flags first ───────────────────────────────
# this must run before any `import jax`
from src.utils.jax_memory_fix_module import apply_jax_memory_fix
# Configure JAX to use 90% of GPU memory with preallocation
apply_jax_memory_fix(fraction=0.9, preallocate=True)
# ─── Now it's safe to pull in JAX (with your flags applied) ─────────────
import jax, jaxlib

import logging
import pymc.sampling.jax as pmjax

import time
from contextlib import contextmanager
from tqdm.auto import tqdm
from jax.lib import xla_bridge
from src.utils.jax_gpu_utils import log_gpu_diagnostics

# Import memory monitoring utilities
from src.utils.jax_memory_monitor import (
    monitor_memory_usage,
    take_memory_snapshot,
    print_memory_snapshot,
    force_allocation_if_needed,
    generate_memory_report
)

# Log GPU diagnostics at startup
log_gpu_diagnostics()

# ── Context manager for timing ─────────────────────────────────────────
@contextmanager
def _timed_section(label: str):
    t0 = time.time()
    yield
    print(f"[{label}] finished in {time.time() - t0:,.1f} s")


# Attempt to import and configure JAX
USE_JAX = True
try:
    # Debug: confirm module and path
    print(f"🔍 JAX module: {jax!r}")
    print(f"🔍 JAX path:   {getattr(jax, '__file__', 'builtin')}")
    if not hasattr(jax, "__version__"):
        raise ImportError("jax.__version__ missing—possible circular import")
    print(f"✅ JAX version: {jax.__version__}")
    jax.config.update("jax_enable_x64", True)
    log_gpu_diagnostics() 
    # ── NEW: Verify PyMC sees GPU backend ─────────────────────────────
    print("🔍 PyMC version:", pm.__version__)
    print("🔍 JAX backend:", xla_bridge.get_backend().platform)

except Exception as e:
    USE_JAX = False
    print(f"⚠️  Warning: could not import/configure JAX ({e}). Falling back to CPU sampling.")


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
    verbose: bool   = True,
    sampler: str    = "jax",
    chains: int     = 1,
    monitor_memory: bool = True,
    force_memory_allocation: bool = True,
    allocation_target: float = 0.8,
    direct_feature_input: tuple = None,  # (X, y, feature_names) for testing
):
    cols = _ColumnSchema()
    if feature_list is None:
        feature_list = cols.model_features()

    # Take initial memory snapshot if monitoring enabled
    if monitor_memory:
        take_memory_snapshot("Before model setup")

    # Force memory allocation if requested
    if force_memory_allocation and monitor_memory:
        if verbose:
            print("\n=== Pre-training Memory Allocation ===")
        allocation_result = force_allocation_if_needed(
            target_fraction=allocation_target,
            current_usage_threshold=0.4,
            step_size_mb=1000,
            max_steps=8,
            verbose=verbose
        )
        if verbose:
            print(f"Memory allocation result: {json.dumps(allocation_result, indent=2)}")

    # 1) design matrix - either from transformer or direct input
    if direct_feature_input is not None:
        # Use direct input (for testing)
        X, y, feature_names = direct_feature_input
        if feature_list is not None:
            # Filter features if list provided
            selected_indices = [i for i, name in enumerate(feature_names) if name in feature_list]
            X = X[:, selected_indices]
            feature_names = [feature_names[i] for i in selected_indices]
    else:
        # Use transformer
        X_all, y_ser = transform_preprocessor(df_raw, transformer)
        names = transformer.get_feature_names_out()
        X = X_all[:, np.isin(names, feature_list)]
        y = y_ser.values
        feature_names = [fn for fn in names if fn in feature_list]

    # Cast indices to Python ints
    n_bat  = int(batter_idx.max()) + 1
    n_lvl  = int(level_idx.max()) + 1
    n_seas = int(season_idx.max()) + 1
    n_pit  = int(pitcher_idx.max()) + 1
    n_feat = int(X.shape[1])

    if verbose:
        print(f"🔍 Data dims: n_bat={n_bat}, n_lvl={n_lvl}, n_seas={n_seas}, n_pit={n_pit}, n_feat={n_feat}")

    with pm.Model() as model:
        # Priors
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

        # Likelihood
        theta = (
            mu
            + beta_level[level_idx]
            + pm.math.dot(X, beta)
            + u[batter_idx]
            + v[season_idx]
            + p[pitcher_idx]
        )
        sigma_e  = pm.HalfNormal("sigma_e", sigma_prior)
        pm.Normal("y_obs", theta, sigma_e, observed=y)

        # Memory snapshot before sampling if monitoring enabled
        if monitor_memory:
            take_memory_snapshot("Before sampling")
            # Take another memory snapshot after a small delay to catch any lazy allocation
            time.sleep(1)
            take_memory_snapshot("Before sampling (after delay)")

        # Sampling with timing and memory monitoring
        sampling_context = (
            monitor_memory_usage("MCMC Sampling", verbose=verbose) if monitor_memory 
            else _timed_section("compile+sample")
        )

        with sampling_context:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                nuts_sampler="numpyro" if (sampler=="jax" and USE_JAX) else "nuts",
                **(
                    {"nuts_sampler_kwargs": {"chain_method": "vectorized"}}
                    if (sampler == "jax" and USE_JAX)
                    else {}
                ),
                progressbar=verbose,
                idata_kwargs={"log_likelihood": ["y_obs"]},
            )

        # Memory snapshot after sampling if monitoring enabled
        if monitor_memory:
            take_memory_snapshot("After sampling")

        # Posterior predictive timing with memory monitoring
        posterior_context = (
            monitor_memory_usage("Posterior Predictive", verbose=verbose) if monitor_memory 
            else _timed_section("posterior_predictive")
        )

        with posterior_context:
            idata.extend(pm.sample_posterior_predictive(idata, var_names=["y_obs"]))

    # Store feature names
    idata.attrs["feature_names"] = feature_names

    if verbose:
        print("⚡ Posterior predictive samples (first 5):",
              idata.posterior_predictive["y_obs"]
                   .stack(s=("chain", "draw")).values[:5])

    # Generate memory report if monitoring enabled
    if monitor_memory:
        report = generate_memory_report("hierarchical_memory_report.json")
        if "summary" in report and "gpu_utilization" in report["summary"]:
            util = report["summary"]["gpu_utilization"]
            print("\n=== Memory Usage Summary ===")
            print(f"GPU Utilization - Min: {util['min']:.2f}% Max: {util['max']:.2f}% Avg: {util['avg']:.2f}%")
            print(f"Detailed report saved to hierarchical_memory_report.json")

    return idata


# ------------------------------------------------------------------





# ───────────────────────────────────────────────────────────────────────
# 6. Smoke test (only run when module executed directly)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    from pathlib import Path
    import pandas as pd, numpy as np, arviz as az
    from src.data.load_data import load_and_clean_data
    from src.features.feature_engineering import feature_engineer
    from src.features.preprocess import (fit_preprocessor,
                                        prepare_for_mixed_and_hierarchical)
    # from src.models.hierarchical import fit_bayesian_hierarchical
    from src.utils.hierarchical_utils import save_model, save_preprocessor
    from src.utils.posterior import posterior_to_frame
    from src.utils.bayesian_metrics import (compute_classical_metrics,
                                            compute_bayesian_metrics,
                                            compute_convergence_diagnostics,
                                            compute_calibration)

    from src.utils.validation import (
        run_kfold_cv_pymc,
        rmse_pymc,
        posterior_predictive_check,
    )
    import json  # Added import for JSON operations

    RAW   = Path("data/Research Data Project/Research Data Project/exit_velo_project_data.csv")
    OUT_NC = Path("data/models/saved_models/exitvelo_hmc.nc")
    OUT_POST = Path("data/models/saved_models/posterior_summary.parquet")
    OUT_PREPROC = Path("data/models/saved_models/preprocessor.joblib")

    # 1 · prep
    df = load_and_clean_data(RAW)
    df_fe = feature_engineer(df)
    df_model = prepare_for_mixed_and_hierarchical(df_fe)

    _, _, tf = fit_preprocessor(df_model, model_type="linear", debug=False)

    b_idx = df_model["batter_id"].cat.codes.values
    l_idx = df_model["level_idx"].values
    s_idx = df_model["season_idx"].values
    p_idx = df_model["pitcher_idx"].values
    draws_and_tune = 50
    target_accept=0.95
    chains=4
    # 2 · fit
    idata = fit_bayesian_hierarchical(
        df_model,
        tf,
        b_idx,            # batter codes
        l_idx,            # level codes
        s_idx,            # season codes
        p_idx,            # pitcher codes
        sampler="jax",
        draws=draws_and_tune,
        tune=draws_and_tune,
        target_accept=target_accept,
        chains=chains,
        monitor_memory=True,  # Enable memory monitoring
        force_memory_allocation=True,  # Force memory allocation
        allocation_target=0.8,  # Target 80% memory utilization
        direct_feature_input=None  # No direct feature input for this example
    )



    idata.attrs["median_age"] = df_model["age"].median()   # ← NEW

    # 3 · persist
    save_model(idata, OUT_NC)
    save_preprocessor(tf, OUT_PREPROC)
    posterior_to_frame(idata).to_parquet(OUT_POST)
    print("✅ training complete – artefacts written:")
    print("   •", OUT_NC)
    print("   •", OUT_POST)
    print("   •", OUT_PREPROC)


    # # ─── NEW: IMPORT VALIDATION UTILITIES ─────────────────────────────
    # # ─── 7) Sanity‐check CV on your real data ───────────────────────────
    # print("\n--- PyMC hierarchical CV on real data ---")
    # cv_scores = run_kfold_cv_pymc(
    #     lambda d: fit_bayesian_hierarchical(
    #         d, tf, b_idx, l_idx, s_idx, p_idx,
    #         sampler="jax",
    #         draws=draws_and_tune,
    #         tune=draws_and_tune,
    #         target_accept=target_accept,
    #         verbose=False,  # silent inside CV
    #         chains = chains
    #     ),
    #     df_model,
    #     k=3
    # )
    # print("CV RMSEs:", cv_scores)


    # Extract the true target vector for classical metrics
    _, y_full = transform_preprocessor(df_model, tf)
    print("=== Classical Metrics ===")
    compute_classical_metrics(idata, y_full.values)

    # print("\n=== Calibration ===")
    # compute_calibration(idata, y_full.values)

    print("\n=== Bayesian Metrics ===")
    compute_bayesian_metrics(idata)

    print("\n=== Convergence Diagnostics ===")
    compute_convergence_diagnostics(idata)


    # # ─── 8) RMSE & PPC on the full training set ─────────────────────────
    # print("\n--- rmse_pymc on full training data ---")
    # rmse_full = rmse_pymc(idata, df_model)
    # print("RMSE (train):", rmse_full)

    # print("\n--- posterior_predictive_check (real data) ---")
    # fig = posterior_predictive_check(idata, df_model, batter_idx=b_idx)
    # fig.savefig("ppc_real_data.png")
    # print("Saved plot to ppc_real_data.png")

