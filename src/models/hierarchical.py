
import pymc as pm
import arviz as az


def fit_bayesian_hierarchical(
    df,
    batter_idx: np.ndarray,
    level_idx: np.ndarray,
    age_centered: np.ndarray,
    mu_prior: float,
    sigma_prior: float,
    use_ar1: bool = False,
    draws: int = 1000,
    tune: int = 1000,
):
    """
    Bayesian hierarchical linear model:

      y_i ~ Normal(mu + β_level[level_idx_i]
                    + β_age * age_centered_i
                    + u[batter_idx_i],
                    sigma_e)

      u ~ Normal(0, sigma_b)         # per-batter random intercepts
      (Or, if use_ar1=True, u follows a random walk.)

    Returns an ArviZ InferenceData with posterior and PPC.
    """
    import pymc as pm, arviz as az
    import numpy as np

    # 1) Observed data
    y = df["exit_velo"].values
    n_bat = df["batter_id"].nunique()
    n_lvl = df["level_idx"].nunique()

    with pm.Model() as model:
        # 2) Hyperpriors
        mu         = pm.Normal("mu", mu_prior, sigma_prior)
        beta_level = pm.Normal("beta_level", 0.0, 5.0, shape=n_lvl)
        beta_age   = pm.Normal("beta_age",   0.0, 1.0)
        sigma_b    = pm.HalfNormal("sigma_b", sigma_prior)
        sigma_e    = pm.HalfNormal("sigma_e", sigma_prior / 2)

        # 3) Random intercepts u_j
        if use_ar1:
            # a Gaussian random walk across the *ordered* batters
            u = pm.GaussianRandomWalk(
                "u", sigma=sigma_b, shape=n_bat
            )
        else:
            u = pm.Normal(
                "u",
                mu=0.0,
                sigma=sigma_b,
                shape=n_bat
            )

        # 4) Per-row linear predictor
        theta = (
            mu
            + beta_level[level_idx]         # level effect per row
            + beta_age * age_centered       # age effect per row
            + u[batter_idx]                 # random intercept per row
        )

        # 5) Likelihood
        y_obs = pm.Normal(
            "y_obs",
            mu=theta,
            sigma=sigma_e,
            observed=y
        )

        # 6) Sample & posterior-predictive
        trace = pm.sample(
            draws=draws,
            tune=tune,
            target_accept=0.9,
            return_inferencedata=True
        )
        ppc = pm.sample_posterior_predictive(
            trace, var_names=["y_obs"]
        )
        idata = az.concat(trace, ppc, join="outer")

    return idata



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
        df_model,
        batter_idx=batter_idx,
        level_idx=level_idx,
        age_centered=age_centered,
        mu_prior=90,
        sigma_prior=5,
        use_ar1=False,
        draws=1000,
        tune=1000
    )
    print(idata)


