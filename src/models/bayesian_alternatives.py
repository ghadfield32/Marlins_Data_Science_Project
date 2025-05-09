
# ─────────────────────────────────────────────────────────────
# src/models/bayesian_alternatives.py
# Implements additional Bayesian engines that all return
# arviz.InferenceData so downstream metrics remain unchanged.
# ─────────────────────────────────────────────────────────────
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import tempfile
import os
from cmdstanpy import CmdStanModel  # Stan program wrapper
import numpy as np
import pandas as pd
import arviz as az
import pyjags   
import tensorflow as tf
import tensorflow_probability as tfp  
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
    Compile a Stan model from source code string and return ArviZ InferenceData.
    This writes the code to a temporary file, compiles it, samples, and cleans up.
    """
    # 1) Write the Stan code to a temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".stan", delete=False
    ) as tmp:
        tmp.write(stan_code)
        stan_file = tmp.name  # Path to pass to CmdStanModel

    try:
        # 2) Instantiate and compile the model from the .stan file
        model = CmdStanModel(
            stan_file=stan_file,
            force_compile=True
        )

        # 3) Draw samples using HMC/NUTS
        fit = model.sample(
            data=stan_data,
            iter_sampling=draws,
            iter_warmup=warmup,
            chains=chains,
            seed=seed
        )

        # 4) Convert to ArviZ InferenceData
        idata = az.from_cmdstanpy(posterior=fit)
        return idata

    finally:
        # 5) Cleanup temp files to prevent clutter
        try:
            os.remove(stan_file)
        except OSError:
            pass



# 2)  PyJAGS (Gibbs)  ----------------------------------------------------
def fit_bayesian_pyjags(
    jags_model: str,
    jags_data: dict,
    *,
    draws: int = 5000,
    burn: int = 1000,
    thin: int = 1,
    chains: int = 4,
    seed: int = 42,
) -> az.InferenceData:
    """
    Fit a JAGS model via PyJAGS and convert to InferenceData.
    """                                               # PyJAGS interface :contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}
    model = pyjags.Model(code=jags_model,
                         data=jags_data,
                         chains=chains,
                         adapt=burn,
                         init=[{".RNG.seed": seed + c * 10} for c in range(chains)])
    samples = model.sample(draws, vars=None, thin=thin)
    idata   = az.from_pyjags(posterior=samples)
    return idata


# 3)  NumPyro (NUTS)  ----------------------------------------------------
def fit_bayesian_numpyro(
    numpyro_model,
    rng_key,
    *,
    draws: int = 1000,
    warmup: int = 500,
    chains: int = 4,
) -> az.InferenceData:
    """
    Run NUTS in NumPyro; return InferenceData (auto-vectorises across chains).
    """
    import jax
    from numpyro.infer import MCMC, NUTS                           # NumPyro HMC/NUTS :contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}
    kernel = NUTS(numpyro_model)
    mcmc   = MCMC(kernel,
                  num_warmup=warmup,
                  num_samples=draws,
                  num_chains=chains,
                  progress_bar=False)
    mcmc.run(rng_key)
    idata  = az.from_numpyro(mcmc)
    return idata


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
    Vanilla single-chain HMC in TensorFlow Probability.
    """                         # TFP HMC kernel :contentReference[oaicite:14]{index=14}:contentReference[oaicite:15]{index=15}
    tfd, tfmcmc = tfp.distributions, tfp.mcmc

    hmc = tfmcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=step_size,
        num_leapfrog_steps=leapfrog_steps)

    adaptive_hmc = tfmcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc,
        num_adaptation_steps=int(0.8 * burnin))

    @tf.function(autograph=False, jit_compile=True)
    def _run_chain():
        return tfmcmc.sample_chain(
            num_results=draws,
            current_state=init_state,
            kernel=adaptive_hmc,
            num_burnin_steps=burnin,
            seed=seed)

    samples, _ = _run_chain()
    # Wrap in dict so ArviZ can coerce
    posterior = {f"param_{i}": s.numpy() for i, s in enumerate(samples)}
    idata     = az.from_dict(posterior=posterior)
    return idata


# 5)  PyMC ADVI (fast VI baseline)  --------------------------------------
def fit_bayesian_pymc_advi(
    pymc_model,
    *,
    draws: int = 1000,
    tune: int = 10000,
) -> az.InferenceData:
    """
    Run Automatic Differentiation VI (ADVI) in PyMC —> InferenceData.
    """
    import pymc as pm                                              # PyMC ADVI API :contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}
    with pymc_model:
        approx = pm.fit(n=tune, method="advi", progressbar=False)  # returns Approximation
        idata  = approx.sample(draws)
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

    # ─── User-configurable sampling parameters ────────────────────────
    N_CHAINS   = 4      # number of MCMC chains (PyMC, Stan, NumPyro, TFP)
    N_DRAWS    = 500   # number of samples to keep per chain
    N_TUNE     = 500   # number of tuning/adaptation steps per chain
    PP_DRAWS   = 250    # draws for posterior_predictive in PyMC-ADVI & attach_ppc
    JAGS_DRAWS = 500   # total samples for PyJAGS
    JAGS_BURN  = 500   # adaptation steps for PyJAGS
    SEED       = 42     # global RNG seed
    # ────────────────────────────────────────────────────────────────────

    # ─── Synthetic data ────────────────────────────────────────────────
    N       = 50
    rng     = np.random.default_rng(SEED)
    x_data  = rng.uniform(-2, 2, size=N)
    y_true  = 1.0 + 2.5 * x_data
    y_obs   = y_true + rng.normal(0, 0.8, size=N)

    # ─── Helper: ensure posterior_predictive["y_obs"] exists ───────────
    def _attach_ppc(idata: az.InferenceData,
                    x: np.ndarray,
                    alpha: str = "alpha",
                    beta: str = "beta",
                    sigma: str = "sigma",
                    draws: int | None = None) -> None:
        """
        Unified, robust method to add posterior predictive samples to idata.
        Ensures a uniform (chain, draw, obs) array is passed to ArviZ.
        """
        post = idata.posterior

        # 1) Extract posterior arrays via stacking
        a = post[alpha].stack(samples=("chain", "draw")).values  # shape: (n_chains*n_draws,)
        b = post[beta].stack(samples=("chain", "draw")).values
        s = post[sigma].stack(samples=("chain", "draw")).values

        # 2) Determine dimensions
        n_chains = post.sizes.get("chain", 1)                  # single‐chain ADVI fallback
        n_draws  = post.sizes["draw"]
        n_obs    = x.shape[0]
        M        = min(a.size, draws) if draws is not None else a.size

        # 3) Build predictive draws (shape: M × n_obs)
        mu  = a[:M, None] + b[:M, None] * x[None, :]
        ypp = rng.normal(loc=mu, scale=s[:M, None])

        # 4) Reshape to (chain, draw, obs)
        arr = ypp.reshape(n_chains, n_draws, n_obs)

        # 5) Attach to InferenceData, letting ArviZ handle chain/draw dims
        idata.add_groups(
            posterior_predictive={"y_obs": arr},
            dims={"y_obs": ["obs"]},
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
        y_obs[i] <- alpha + beta*x[i] + sigma*randn();
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
    import numpyro, numpyro.distributions as dist
    def numpyro_model(x, y=None):
        a = numpyro.sample("alpha", dist.Normal(0,5))
        b = numpyro.sample("beta",  dist.Normal(0,5))
        s = numpyro.sample("sigma", dist.HalfNormal(1))
        mu = a + b * x
        numpyro.sample("y", dist.Normal(mu, s), obs=y)
    rng_key = jr.PRNGKey(SEED)
    idata_numpyro = fit_bayesian_numpyro(numpyro_model,
                                         rng_key,
                                         draws=N_DRAWS,
                                         warmup=N_TUNE,
                                         chains=N_CHAINS)
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
    _attach_ppc(idata_tfp, x_data, sigma="param_2", draws=PP_DRAWS)

    # ─── Compare & Plot ───────────────────────────────────────────────
    from src.utils.bayesian_metrics import compute_classical_metrics
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
        a = id_.posterior["alpha"].mean().item()
        b = id_.posterior["beta"].mean().item()
        plt.plot(line_x, a + b*line_x, color=c, label=name, lw=1.2)
    plt.legend(); plt.title("Posterior mean fits – smoke test")
    plt.tight_layout()
    Path("smoke_test_fits.png").write_bytes(plt.savefig("/tmp/_smoke.png") or b"")
    print("\n✔︎ Smoke-test figure saved → smoke_test_fits.png")
