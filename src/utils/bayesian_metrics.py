import arviz as az
from sklearn.metrics import mean_squared_error,root_mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def compute_classical_metrics(idata, y_true):
    """Compute MSE, RMSE, MAE & R² from the posterior predictive distribution."""
    # Extract posterior predictive draws and compute mean prediction
    y_ppc = (
        idata.posterior_predictive['y_obs']
        .stack(samples=('chain', 'draw'))
        .values
    )
    y_pred = y_ppc.mean(axis=1)

    # Classical regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Print results
    print(f"▶ Classical MSE : {mse:.2f}")
    print(f"▶ Classical RMSE: {rmse:.2f}")
    print(f"▶ Classical MAE : {mae:.2f}")
    print(f"▶ Classical R²  : {r2:.3f}")

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}





def _chunked_loo_waic(idata, chunk_size):
    """
    Internal helper: compute pointwise LOO/WAIC in chunks of observations.
    """
    # Extract raw log-likelihood array [chains, draws, obs]
    ll = idata.log_likelihood["y_obs"].values
    n_obs = ll.shape[-1]

    elpd_loo_i   = np.empty(n_obs, dtype=float)
    p_loo_i      = np.empty(n_obs, dtype=float)
    pareto_k_all = []
    elpd_waic_i  = np.empty(n_obs, dtype=float)
    p_waic_i     = np.empty(n_obs, dtype=float)

    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        subset = idata.isel(observed_data={"obs": slice(start, end)})

        loo_sub  = az.loo(subset,  pointwise=True)
        waic_sub = az.waic(subset, pointwise=True)

        elpd_loo_i [start:end] = loo_sub["elpd_loo"].values
        p_loo_i    [start:end] = loo_sub["p_loo"].values
        pareto_k_all.extend(loo_sub["pareto_k"].values.tolist())

        elpd_waic_i[start:end] = waic_sub["elpd_waic"].values
        p_waic_i   [start:end] = waic_sub["p_waic"].values

    loo = az.ELPDData(
        {"elpd_loo": ("obs", elpd_loo_i),
         "p_loo":    ("obs", p_loo_i),
         "pareto_k": ("obs", np.array(pareto_k_all))},
        coords={"obs": np.arange(n_obs)},
        dims   ={"elpd_loo": ["obs"], "p_loo": ["obs"], "pareto_k": ["obs"]}
    )
    waic = az.ELPDData(
        {"elpd_waic": ("obs", elpd_waic_i),
         "p_waic":    ("obs", p_waic_i)},
        coords={"obs": np.arange(n_obs)},
        dims   ={"elpd_waic": ["obs"], "p_waic": ["obs"]}
    )
    return loo, waic


def compute_bayesian_metrics(idata, *, use_chunks: bool = False, chunk_size: int | None = None):
    """Compute PSIS-LOO, WAIC and Pareto k diagnostics for model comparison."""
    # 1) Guard missing log_likelihood
    if 'log_likelihood' not in idata.groups():
        print("⚠️ InferenceData has no log_likelihood; skipping LOO/WAIC.")
        return None

    # Validate chunk parameters
    if use_chunks:
        if chunk_size is None or chunk_size <= 0:
            raise ValueError("When use_chunks=True, chunk_size must be a positive integer")

    # 2) Compute LOO & WAIC (batch or chunked)
    try:
        if not use_chunks:
            loo  = az.loo(idata,  pointwise=True)
            waic = az.waic(idata, pointwise=True)
        else:
            loo, waic = _chunked_loo_waic(idata, chunk_size)
    except Exception as e:
        print(f"⚠️ LOO/WAIC computation failed: {e}")
        return None

    # 3) Extract scalars
    looic    = -2 * loo["elpd_loo"].sum()
    p_loo    = loo["p_loo"].sum()
    waic_val = -2 * waic["elpd_waic"].sum()
    p_waic   = waic["p_waic"].sum()
    prop_bad_k = np.mean(loo["pareto_k"].values > 0.7)

    # 4) Print results
    print(f"▶ LOOIC       : {looic:.1f}, p_loo: {p_loo:.1f}")
    print(f"▶ WAIC        : {waic_val:.1f}, p_waic: {p_waic:.1f}")
    print(f"▶ Pareto k>0.7: {prop_bad_k:.2%} of observations")

    return {'loo': loo, 'waic': waic, 'prop_bad_k': prop_bad_k}





def compute_convergence_diagnostics(idata):
    """Print R̂ and ESS bulk/tail for key parameters."""
    if 'posterior' not in idata.groups():
        print("No posterior group in InferenceData; skipping convergence diagnostics.")
        return None

    try:
        summary = az.summary(
            idata,
            var_names=["mu", "beta", "sigma_e"],
            kind="diagnostics",
            round_to=2
        )
    except KeyError as e:
        print(f"Convergence diagnostics skipped: missing vars {e}")
        return None

    print("▶ Convergence diagnostics (R̂, ESS):")
    print(summary[["r_hat", "ess_bulk", "ess_tail"]])
    return summary


def compute_calibration(idata, y_true, hdi_prob=0.95):
    """
    Compute calibration: the fraction of true observations that lie within the posterior predictive HDI.
    """
    # Extract posterior predictive draws
    y_ppc = (
        idata.posterior_predictive['y_obs']
        .stack(samples=('chain','draw'))
        .values
    )
    # Compute lower/upper quantiles per observation
    lower = np.percentile(y_ppc, (1-hdi_prob)/2*100, axis=1)
    upper = np.percentile(y_ppc, (1+(hdi_prob))/2*100, axis=1)
    within = ((y_true >= lower) & (y_true <= upper)).mean()
    print(f"▶ Calibration: {within:.2%} of true values within {int(hdi_prob*100)}% HDI")
    return within

if __name__ == "__main__":
    import numpy as np
    import arviz as az

    # 1) True values
    np.random.seed(42)
    y_true = np.random.normal(0, 1, size=10)

    # 2) Posterior predictive draws (2 chains × 5 draws × 10 obs)
    y_obs = np.random.normal(loc=y_true, scale=0.5, size=(2,5,10))

    # 3) Posterior parameters μ, β, σₑ (2 chains × 5 draws)
    posterior = {
        'mu':      np.random.normal(0, 1, size=(2,5)),
        'beta':    np.random.normal(0, 1, size=(2,5)),
        'sigma_e': np.abs(np.random.normal(1, 0.2, size=(2,5))),
    }

    # 4) Log‑likelihood (for each chain, draw, obs)
    log_lik = np.random.normal(-1, 0.5, size=(2,5,10))

    # 5) Build InferenceData
    idata = az.from_dict(
        posterior=posterior,
        posterior_predictive={'y_obs': y_obs},
        log_likelihood     ={'y_obs': log_lik},
        dims={
            'y_obs': ['chain','draw','obs'],
            'mu':    ['chain','draw'],
            'beta':  ['chain','draw'],
            'sigma_e':['chain','draw'],
        },
        coords={
            'chain': [0,1],
            'draw':  list(range(5)),
            'obs':   list(range(10))
        }
    )

    print("=== Classical Metrics ===")
    compute_classical_metrics(idata, y_true)

    print("\n=== Calibration ===")
    compute_calibration(idata, y_true)

    print("\n=== Bayesian Metrics ===")
    compute_bayesian_metrics(idata)

    print("\n=== Convergence Diagnostics ===")
    compute_convergence_diagnostics(idata)
