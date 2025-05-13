

# ─────────────────────────────────────────────────────────────
# src/models/bayesian_alternatives.py
# Implements additional Bayesian engines that all return
# arviz.InferenceData so downstream metrics remain unchanged.
# ─────────────────────────────────────────────────────────────

# ─── Configure JAX/GPU ──────────────────────────────────────────────
from src.utils.jax_memory_fix_module import apply_jax_memory_fix
apply_jax_memory_fix(fraction=0.10, preallocate=False)
import jax
jax.config.update("jax_enable_x64", True)

# ─── Timing helper ─────────────────────────────────────────────────
import time
from contextlib import contextmanager

@contextmanager
def _timed_section(label: str):
    t0 = time.perf_counter()
    yield
    print(f"[{label}] done in {time.perf_counter() - t0:.2f}s")

from pathlib import Path
from typing import Tuple
import tempfile
import os
from cmdstanpy import CmdStanModel
import numpy as np
import pandas as pd
import pyjags
import tensorflow as tf
import tensorflow_probability as tfp
import json
from pathlib import Path
import arviz as az
import matplotlib.pyplot as plt

def save_posterior_summary(
    idata: az.InferenceData,
    roster_df: pd.DataFrame,
    output_parquet: Path,
    u_var: str = "u"
) -> pd.DataFrame:
    """
    Extract per-batter random effect 'u' quantiles and save to Parquet.
    Ensures that, even if u_var is missing, an (empty) parquet is written.
    """
    import logging
    # ensure directory exists
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    posterior_vars = set(idata.posterior.data_vars)
    if u_var not in posterior_vars:
        logging.getLogger(__name__).warning(
            "save_posterior_summary: variable %r not found in posterior vars %s; writing empty summary.",
            u_var, sorted(posterior_vars)
        )
        # empty with expected schema
        empty = pd.DataFrame(columns=[
            "batter_id", "batter_idx", "u_q2.5", "u_q50", "u_q97.5"
        ])
        empty.to_parquet(output_parquet, index=False)
        return empty

    # proceed normally
    summary = (
        az.summary(idata, var_names=[u_var], hdi_prob=0.95)
        .reset_index()
        .rename(columns={
            "mean":     "u_q50",
            "hdi_2.5%": "u_q2.5",
            "hdi_97.5%":"u_q97.5"
        })
    )
    df = (
        roster_df[["batter_id", "batter_idx"]]
        .merge(summary, left_on="batter_idx", right_on="index", how="left")
        .loc[:, ["batter_id", "batter_idx", "u_q2.5", "u_q50", "u_q97.5"]]
    )
    df.to_parquet(output_parquet, index=False)
    return df



def save_global_effects(
    idata: az.InferenceData,
    training_df: pd.DataFrame,
    output_json: Path,
    *,
    age_col: str = "age",
    level_idx: int = 2,
    var_alias: dict | None = None,      # NEW
) -> dict | None:
    """
    Extract global intercept & slopes. If the model does not contain the
    required variables, emit *None* and skip saving.
    """
    import json, logging
    log = logging.getLogger(__name__)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # --- 1) Resolve variable names ---------------------------------------
    default_alias = {"Intercept": "alpha", "beta_age": "beta_age",
                     "beta_level": "beta_level"}
    alias = {**default_alias, **(var_alias or {})}

    posterior = idata.posterior
    try:
        mu_mean    = posterior[alias["Intercept"]].mean().item()
        beta_age   = posterior[alias["beta_age"]].mean().item()
        beta_level = (
            posterior[alias["beta_level"]][..., level_idx].mean().item()
        )
    except KeyError as err:
        log.warning(
            "save_global_effects → missing %s in posterior groups %s – "
            "skipping global-effects write for this engine.",
            err, list(posterior.data_vars)
        )
        return None                         # ← early-exit

    glob = {
        "mu_mean":    mu_mean,
        "beta_age":   beta_age,
        "beta_level": {str(level_idx): beta_level},
        "median_age": float(training_df[age_col].median()),
    }
    output_json.write_text(json.dumps(glob, indent=2))
    return glob


def summarize_coefficients(idata, var_names=None):
    """
    Return a DataFrame of mean + 95% HDI for selected variables.
    """
    df = az.summary(idata, var_names=var_names, hdi_prob=0.95)[
        ['mean','hdi_2.5%','hdi_97.5%']
    ]
    return df

def plot_forest_coefficients(idata, var_names=None, figsize=(8,6)):
    """
    Show a forest plot for the specified coefficients.
    """
    plt.figure(figsize=figsize)
    az.plot_forest(idata, var_names=var_names, combined=True, credible_interval=0.95)
    plt.tight_layout()
    plt.show()

def plot_posterior_distributions(idata, var_names=None):
    """
    Show posterior density plots for selected variables.
    """
    az.plot_posterior(idata, var_names=var_names, hdi_prob=0.95)
    plt.tight_layout()
    plt.show()






# 1)  CmdStanPy  ---------------------------------------------------------
def fit_bayesian_cmdstanpy(
    stan_code: str,
    stan_data: dict,
    *,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    seed: int = 42,
) -> az.InferenceData:
    """
    Compile Stan code, run sampling with CmdStanPy, and return ArviZ InferenceData
    with both 'posterior' and 'posterior_predictive' groups.
    """
    with _timed_section("fit_bayesian_cmdstanpy"):
        # Write Stan code to a temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".stan", delete=False) as tmp:
            tmp.write(stan_code)
            stan_file = tmp.name

        try:
            # Compile & sample
            model = CmdStanModel(stan_file=stan_file, force_compile=True)
            fit   = model.sample(
                data=stan_data,
                iter_sampling=draws,
                iter_warmup=warmup,
                chains=chains,
                seed=seed
            )

            # Convert → InferenceData, specifying y_obs as the predictive variable
            idata = az.from_cmdstanpy(
                posterior=fit,
                posterior_predictive=["y_obs"]  # ✔️ correct usage
            )
        finally:
            # Cleanup temp file
            try:
                os.remove(stan_file)
            except OSError:
                pass

    return idata



# 2)  PyJAGS (Gibbs)  ----------------------------------------------------
# ─────────────────────────────────────────────────────────────
# UPDATED: fit_bayesian_pyjags  (drop-in replacement)
# ─────────────────────────────────────────────────────────────
import tempfile, os, arviz as az, pyjags, itertools
from contextlib import contextmanager
from typing import Sequence, Optional, List

# Valid RNG factories supported by JAGS’ base / lecuyer modules
VALID_RNGS: List[str] = [
    "base::Wichmann-Hill",
    "base::Marsaglia-Multicarry",
    "base::Super-Duper",
    "base::Mersenne-Twister",
    "lecuyer::RngStream",
]

def fit_bayesian_pyjags(
    jags_model: str,
    jags_data: dict,
    *,
    draws: int = 5_000,
    burn: int = 1_000,
    thin: int = 1,
    chains: int = 4,
    seed: int = 42,
    rng_name: Optional[str] = None,
) -> az.InferenceData:
    """
    Fit a BUGS model in JAGS via PyJAGS and return ArviZ InferenceData.

    Parameters
    ----------
    rng_name : str | None
        One of the names in `VALID_RNGS.  If None (default) the function
        rotates valid RNGs across chains so every chain uses an independent
        generator, per JAGS best-practice.
    """
    # 1) Write BUGS code to a temporary file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".bug", delete=False)
    try:
        tmp.write(jags_model)
        tmp.close()
        print(f"💾 JAGS model written to {tmp.name}")

        # 2) Choose RNGs for each chain
        if rng_name is not None:
            if rng_name not in VALID_RNGS:
                raise ValueError(
                    f"rng_name must be one of {VALID_RNGS}, got {rng_name!r}"
                )
            rngs: Sequence[str] = [rng_name] * chains
        else:
            # Rotate through the supported generators
            rngs = list(itertools.islice(itertools.cycle(VALID_RNGS), chains))

        # 3) Initial-value dicts (independent seeds per chain)
        init_vals = [
            {".RNG.name": rngs[c],
             ".RNG.seed": seed + c * 10}
            for c in range(chains)
        ]

        # 4) Build and adapt the model
        with _timed_section("fit_bayesian_pyjags"):
            model = pyjags.Model(
                file=tmp.name,
                data=jags_data,
                chains=chains,
                adapt=burn,
                init=init_vals
            )

            # 5) Draw samples
            samples = model.sample(draws, vars=None, thin=thin)

        # 6) Convert → InferenceData
        return az.from_pyjags(posterior=samples)

    finally:
        # Always clean up temp file
        try:
            os.remove(tmp.name)
        except OSError:
            pass




# 3)  NumPyro (NUTS)  ----------------------------------------------------
def fit_bayesian_numpyro(
    numpyro_model,
    rng_key,
    *model_args,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
    progress_bar: bool = False,
    **model_kwargs
) -> az.InferenceData:
    """
    Run NUTS in NumPyro and return ArviZ InferenceData.

    Parameters
    ----------
    numpyro_model : callable
        A NumPyro model accepting positional args and kwargs (e.g., x, y=y).
    rng_key : jax.random.PRNGKey
        Random number generator key for reproducibility.
    *model_args : tuple
        Positional arguments to pass into the model (e.g., x_data).
    draws : int
        Number of posterior samples per chain.
    warmup : int
        Number of warmup (tuning) iterations.
    chains : int
        Number of MCMC chains.
    progress_bar : bool
        Whether to show a progress bar during sampling.
    **model_kwargs : dict
        Keyword arguments to pass into the model (e.g., y=y_obs).

    Returns
    -------
    idata : arviz.InferenceData
        InferenceData containing posterior samples and diagnostics.
    """
    import numpyro
    from numpyro.infer import MCMC, NUTS

    # 1) Instantiate the NUTS kernel and MCMC controller
    kernel = NUTS(numpyro_model)
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=draws,
        num_chains=chains,
        progress_bar=progress_bar
    )

    # 2) Run sampling, passing model_args and model_kwargs
    #    model_args and model_kwargs map to NumPyro's model_args/model_kwargs
    mcmc.run(rng_key, *model_args, **model_kwargs)  # :contentReference[oaicite:3]{index=3}

    # 3) Convert to ArviZ InferenceData and return
    return az.from_numpyro(mcmc)


# 4)  TensorFlow Probability HMC  ---------------------------------------
def fit_bayesian_tfp_hmc(
    target_log_prob_fn,
    init_state,
    *,
    step_size: float = 0.05,
    leapfrog_steps: int = 5,
    draws: int = 1000,
    burnin: int = 500,
    seed: int = 42,
) -> az.InferenceData:
    """
    Single-chain HMC in TensorFlow Probability, returning posterior
    variables named 'alpha', 'beta', 'log_sigma', and 'sigma'.
    """
    import tensorflow as tf
    import tensorflow_probability as tfp
    import numpy as np

    tfd, tfmcmc = tfp.distributions, tfp.mcmc

    with _timed_section("fit_bayesian_tfp_hmc"):
        # Build & adapt the HMC kernel
        hmc = tfmcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=leapfrog_steps,
        )
        adaptive = tfmcmc.SimpleStepSizeAdaptation(
            inner_kernel=hmc,
            num_adaptation_steps=int(0.8 * burnin),
        )

        # Run the chain
        @tf.function(autograph=False, jit_compile=True)
        def _run_chain():
            return tfmcmc.sample_chain(
                num_results=draws,
                current_state=init_state,
                kernel=adaptive,
                num_burnin_steps=burnin,
                seed=seed,
            )

        # Unpack raw states: (alpha, beta, log_sigma)
        (alpha_t, beta_t, log_sigma_t), _ = _run_chain()

    # To NumPy
    alpha_np     = alpha_t.numpy()
    beta_np      = beta_t.numpy()
    log_sigma_np = log_sigma_t.numpy()
    sigma_np     = np.exp(log_sigma_np)  # transform

    # Build posterior dict with both raw and transformed sigma
    posterior = {
        "alpha":     alpha_np,
        "beta":      beta_np,
        "log_sigma": log_sigma_np,  # keep raw chain
        "sigma":     sigma_np,      # transformed for usability
    }

    # Return as ArviZ InferenceData
    return az.from_dict(posterior=posterior)




# 5)  PyMC ADVI (fast VI baseline)  --------------------------------------
def fit_bayesian_pymc_advi(
    pymc_model,
    *,
    draws: int = 1_000,
    tune:  int = 10_000,
    progressbar: bool = False,
) -> az.InferenceData:
    """
    Run Automatic Differentiation VI (ADVI) in PyMC and return InferenceData.

    * Works with both PyMC <5 and ≥5.
    * Falls back gracefully if a custom start dict is invalid.
    * Emits timing + initial-point diagnostics for reproducibility.
    """
    import pymc as pm
    import logging

    logger = logging.getLogger(__name__)

    with _timed_section("fit_bayesian_pymc_advi"):
        with pymc_model:
            # ------------------------------------------------------------------
            # 1) Obtain a *numeric* initial point compatible with the PyMC core
            # ------------------------------------------------------------------
            start = None
            try:
                iprop = pymc_model.initial_point      # may be dict *or* callable
                start = iprop() if callable(iprop) else iprop
                if not isinstance(start, dict):
                    raise TypeError("initial_point did not return a dict.")
                logger.info(
                    "Using model.initial_point (keys=%s...)",
                    list(start)[:4]
                )
            except Exception as err:  # noqa: BLE001
                logger.warning(
                    "Falling back to PyMC auto-initialisation (err=%s)", err
                )
                start = None          # Let pm.fit decide

            # ------------------------------------------------------------------
            # 2) Fit ADVI
            # ------------------------------------------------------------------
            approx = pm.fit(
                n=tune,
                method="advi",
                start=start,
                progressbar=progressbar,
            )

            # ------------------------------------------------------------------
            # 3) Draw posterior samples & convert to InferenceData
            # ------------------------------------------------------------------
            idata = approx.sample(draws)

    return idata



# ───────────────────────────────────────────────────────────────────────
# 6. Smoke test (only run when module executed directly)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import jax.random as jr
    import tensorflow_probability as tfp
    import tensorflow as tf
    import pymc as pm
    from cmdstanpy import CmdStanModel
    import arviz as az
    import matplotlib.pyplot as plt
    from pathlib import Path
    import numpy as np
    import pandas as pd

    # ─── Additional imports for summaries & interpretability ────────────
    from src.models.hierarchical_predict import (
        predict_from_summaries,
        get_top_hitters
    )
    from src.features.feature_engineering import feature_engineer
    import numpyro, numpyro.distributions as dist
    from src.utils.bayesian_metrics import compute_classical_metrics
    # ─── User-configurable sampling parameters ──────────────────────────
    N_CHAINS   = 4
    N_DRAWS    = 500
    N_TUNE     = 500
    PP_DRAWS   = 250
    JAGS_DRAWS = 500
    JAGS_BURN  = 500
    SEED       = 42
    # ────────────────────────────────────────────────────────────────────

    # ─── Synthetic data ────────────────────────────────────────────────
    N       = 50
    rng     = np.random.default_rng(SEED)
    x_data  = rng.uniform(-2, 2, size=N)
    y_true  = 1.0 + 2.5 * x_data
    y_obs   = y_true + rng.normal(0, 0.8, size=N)

    # ─── Helper to attach posterior predictive draws ───────────────────
    def _attach_ppc(idata, x, alpha="alpha", beta="beta", sigma="sigma", draws=None):
        post = idata.posterior
        a_flat = post[alpha].stack(samples=("chain","draw")).values
        b_flat = post[beta ].stack(samples=("chain","draw")).values
        s_flat = post[sigma].stack(samples=("chain","draw")).values

        n_chains = post.sizes.get("chain", 1)
        n_draws_param = post.sizes["draw"]
        n_obs = x.shape[0]

        if draws is not None:
            total = min(a_flat.size, draws * n_chains)
            per_chain = total // n_chains
        else:
            total = a_flat.size
            per_chain = n_draws_param

        mu = a_flat[:total, None] + b_flat[:total, None] * x[None, :]
        ypp = rng.normal(loc=mu, scale=s_flat[:total, None])
        arr = ypp.reshape(n_chains, per_chain, n_obs)

        idata.add_groups(
            posterior_predictive={"y_obs": arr},
            dims={"y_obs": ["obs"]}
        )




    # ─── 1) PyMC HMC (NUTS) ───────────────────────────────────────────
    coords = {"obs": np.arange(N)}
    with pm.Model(coords=coords) as pymc_model:
        alpha = pm.Normal("alpha", 0, 5)
        beta  = pm.Normal("beta",  0, 5)
        sigma = pm.HalfNormal("sigma", 1)
        mu    = alpha + beta * x_data
        pm.Normal("y", mu, sigma, observed=y_obs, dims="obs")

    idata_pymc_hmc = pm.sample(
        N_DRAWS,
        tune=N_TUNE,
        chains=N_CHAINS,
        random_seed=SEED,
        progressbar=False,
        return_inferencedata=True,
        model=pymc_model
    )
    pm.sample_posterior_predictive(
        idata_pymc_hmc,
        var_names=["y"],
        extend_inferencedata=True,
        model=pymc_model
    )
    idata_pymc_hmc.posterior_predictive = \
        idata_pymc_hmc.posterior_predictive.rename({"y": "y_obs"})

    # ─── 2) PyMC ADVI ─────────────────────────────────────────────────
    idata_pymc_advi = fit_bayesian_pymc_advi(pymc_model,
                                             draws=PP_DRAWS,
                                             tune=N_TUNE*5)
    _attach_ppc(idata_pymc_advi, x_data, draws=PP_DRAWS)

    # ─── 3) CmdStanPy (Stan HMC) ──────────────────────────────────────
    stan_code = """
    data { int<lower=0> N; vector[N] x; vector[N] y; }
    parameters { real alpha; real beta; real<lower=0> sigma; }
    model { y ~ normal(alpha+beta*x, sigma); }
    generated quantities {
      vector[N] y_obs;
      for (n in 1:N)
        y_obs[n] = normal_rng(alpha+beta*x[n], sigma);
    }
    """
    stan_data = {"N": N, "x": x_data, "y": y_obs}
    idata_stan = fit_bayesian_cmdstanpy(stan_code,
                                        stan_data,
                                        draws=N_DRAWS,
                                        warmup=N_TUNE,
                                        chains=N_CHAINS,
                                        seed=SEED)

    # ─── 4) PyJAGS (Gibbs Sampling) ───────────────────────────────────
    jags_model = """
    model {
      for (i in 1:N) {
        y[i] ~ dnorm(alpha+beta*x[i], tau);
        y_obs[i] ~ dnorm(alpha + beta*x[i], tau);

      }
      alpha ~ dnorm(0, .001);
      beta  ~ dnorm(0, .001);
      sigma ~ dunif(0, 10);
      tau   <- pow(sigma, -2);
    }
    """
    jags_data = {"N": N, "x": x_data, "y": y_obs}
    idata_jags = fit_bayesian_pyjags(jags_model,
                                     jags_data,
                                     draws=JAGS_DRAWS,
                                     burn=JAGS_BURN,
                                     chains=N_CHAINS,
                                     seed=SEED)
    _attach_ppc(idata_jags, x_data, draws=PP_DRAWS)

    # ─── 5) NumPyro (NUTS) ─────────────────────────────────────────────
    def numpyro_model(x, y=None):
        a = numpyro.sample("alpha", dist.Normal(0,5))
        b = numpyro.sample("beta",  dist.Normal(0,5))
        s = numpyro.sample("sigma", dist.HalfNormal(1))
        mu = a + b * x
        numpyro.sample("y", dist.Normal(mu, s), obs=y)
    rng_key = jr.PRNGKey(SEED)
    idata_numpyro = fit_bayesian_numpyro(
        numpyro_model,
        rng_key,
        x_data,                # positional model argument
        y=y_obs,               # keyword model argument
        draws=N_DRAWS,
        warmup=N_TUNE,
        chains=N_CHAINS
    )

    _attach_ppc(idata_numpyro, x_data, draws=PP_DRAWS)

    # ─── 6) TFP HMC ───────────────────────────────────────────────────
    tfd, tfmcmc = tfp.distributions, tfp.mcmc
    def target_log_prob_fn(alpha, beta, log_sigma):
        sigma = tf.exp(log_sigma)
        yhat  = alpha + beta * tf.constant(x_data, tf.float32)
        return tf.reduce_sum(tfd.Normal(yhat, sigma).log_prob(y_obs)) + \
               tfd.Normal(0,5).log_prob(alpha) + \
               tfd.Normal(0,5).log_prob(beta) + \
               tfd.Normal(0,1).log_prob(log_sigma)

    init_state = [tf.zeros([]), tf.zeros([]), tf.zeros([])]
    idata_tfp  = fit_bayesian_tfp_hmc(target_log_prob_fn,
                                      init_state,
                                      draws=N_DRAWS,
                                      burnin=N_TUNE,
                                      seed=SEED)

    # 2) Attach posterior predictive draws with positive sigma
    _attach_ppc(
        idata_tfp,
        x_data,
        alpha="alpha",
        beta="beta",
        sigma="sigma",
        draws=PP_DRAWS
    )

    # ─── Compare & Plot ───────────────────────────────────────────────
    engines = {
      "PyMC-HMC":  idata_pymc_hmc,
      "PyMC-ADVI": idata_pymc_advi,
      "Stan":      idata_stan,
      "JAGS":      idata_jags,
      "NumPyro":   idata_numpyro,
      "TFP-HMC":   idata_tfp,
    }
    for name,idata in engines.items():
        print(f"\n▼▼ {name}")
        compute_classical_metrics(idata, y_obs)

    # quick visual check (posterior mean lines)
    plt.figure(figsize=(6,4))
    plt.scatter(x_data, y_obs, label="obs", alpha=.6)
    line_x = np.linspace(-2, 2, 100)
    colors = plt.cm.tab10(np.linspace(0,1,len(engines)))
    for c,(name,id_) in zip(colors, engines.items()):
        a = idata_tfp.posterior["alpha"].mean().item()
        b = idata_tfp.posterior["beta"].mean().item()

        plt.plot(line_x, a + b*line_x, color=c, label=name, lw=1.2)
    plt.legend(); plt.title("Posterior mean fits – smoke test")
    plt.tight_layout()
    Path("data/images/smoke_test_fits.png").write_bytes(plt.savefig("/tmp/_smoke.png") or b"")
    print("\n✔︎ Smoke-test figure saved → smoke_test_fits.png")


    # ─── Prepare real roster & training data for prediction ───────────
    BASE = Path("data/models")
    ROSTER = Path("data/Research Data Project/Research Data Project/exit_velo_validate_data.csv")
    RAW    = Path("data/Research Data Project/Research Data Project/exit_velo_project_data.csv")
    # 2) run hierarchical exit‐velo pipeline

    roster_df   = pd.read_csv(ROSTER)
    training_df = feature_engineer(pd.read_csv(RAW))

    engines = {
      "Stan":      idata_stan,
      "JAGS":      idata_jags,
      "NumPyro":   idata_numpyro,
      "TFP-HMC":   idata_tfp,
      "PyMC-HMC":  idata_pymc_hmc,
      "PyMC-ADVI": idata_pymc_advi,
    }

    for name, idata in engines.items():
        summary_file = BASE/name/"posterior_summary.parquet"
        globals_file = BASE/name/"global_effects.json"

        save_posterior_summary(idata, roster_df, summary_file, u_var="u")
        # 1) Save globals; skip this engine if it returns None
        glob_res = save_global_effects(
            idata, training_df, globals_file, age_col="age"
        )
        if glob_res is None:
            print(f"[Warning] no global-effects for '{name}', skipping predictions")
            continue

        # 2) Attempt predictions; catch file‑or‑schema errors and skip
        try:
            df_pred = predict_from_summaries(
                roster_csv=ROSTER,
                raw_csv=RAW,
                posterior_parquet=summary_file,
                global_effects_json=globals_file,
                output_csv=BASE/name/"predictions_2024.csv",
                verbose=True
            )
        except ValueError as err:
            print(f"[Error] skipping '{name}': {err}")
            continue

        # 3) If we got this far, we have predictions → extract top hitters
        get_top_hitters(df_pred, hitter_col="hitter_type", n=5, verbose=True)


