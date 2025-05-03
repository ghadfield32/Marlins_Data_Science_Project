
import pymc as pm
import arviz as az


def fit_bayesian_hierarchical(df,
                              batter_idx,
                              level_idx,
                              age_centered,
                              mu_prior: float,
                              sigma_prior: float,
                              use_ar1: bool = False,
                              draws: int = 1000,
                              tune: int = 1000):
    """
    Exit-velo model with level & age effects.

    y_i ~ Normal(theta_i, σ_e)
    theta_i = μ + β_level[level_idx_i] + β_age * age_centered_i + η_i

    If `use_ar1=True`, η_i follows a Gaussian random walk
    sorted by season (requires season order in caller).

    Returns ArviZ InferenceData (posterior + PPC).
    """
    y = df["exit_velo"].values
    n_bat = df["batter_id"].nunique()
    n_lvl = df["level_idx"].nunique()

    with pm.Model():
        mu = pm.Normal("mu", mu_prior, sigma_prior)
        beta_level = pm.Normal("beta_level", 0.0, 5.0, shape=n_lvl)
        beta_age = pm.Normal("beta_age", 0.0, 1.0)
        sigma_b = pm.HalfNormal("sigma_b", sigma_prior)
        sigma_e = pm.HalfNormal("sigma_e", sigma_prior / 2)

        theta_mean = (
            mu
            + beta_level[level_idx]
            + beta_age * age_centered
        )

        if use_ar1:
            theta = pm.GaussianRandomWalk("theta", sigma=sigma_b, shape=n_bat)
        else:
            theta = pm.Normal("theta", theta_mean, sigma_b, shape=n_bat)

        _y_obs = pm.Normal("y_obs", theta[batter_idx], sigma_e, observed=y)

        trace = pm.sample(draws=draws, tune=tune,
                          target_accept=0.9,
                          return_inferencedata=True)
        ppc = pm.sample_posterior_predictive(trace,
                                             var_names=["y_obs"])
        idata = az.concat(trace, ppc, join="outer")

    return idata


# ───────────────────────────────────────────────────────────────────────
# 6. Smoke test (only run when module executed directly)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_raw
    from src.features.feature_engineering import feature_engineer

    raw_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
    df = load_raw(raw_path)
    print(df.head())
    print(df.columns)

    # --- inspect nulls in the raw data ---
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print("✅  No missing values in raw data.")
    else:
        print("=== Raw data null counts ===")
        for col, cnt in null_counts.items():
            print(f" • {col!r}: {cnt} missing")
    df_fe = feature_engineer(df)

    print("Raw →", df.shape, "//  Feature‑engineered →", df_fe.shape)
    print(df_fe.head())

    # singleton instance people can import as `cols`
    cols = _ColumnSchema()

    __all__ = ["cols"]
    print("ID columns:         ", cols.id())
    print("Ordinal columns:    ", cols.ordinal())
    print("Nominal columns:    ", cols.nominal())
    print("All categorical:    ", cols.categorical())
    print("Numerical columns:  ", cols.numerical())
    print("Model features:     ", cols.model_features())
    print("Target columns:  ", cols.target())
    print("All raw columns:    ", cols.all_raw())
    numericals = cols.numerical()
    # use list‐comprehension to drop target(s) from numerical features
    numericals_without_y = [c for c in numericals if c not in cols.target()]

    summary_df = summarize_categorical_missingness(df_fe)
    print(summary_df.to_markdown(index=False))


    # check nulls
    print("🛠️  Nulls in X before fit_transform:")
    null_counts = df_fe.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print("✅  No missing values after feature engineering.")
    else:
        print("=== Null counts post-engineering ===")
        print(null_counts)

    train_df, test_df = train_test_split(df_fe, test_size=0.2, random_state=42)

    # run with debug prints
    X_train, y_train, tf = fit_preprocessor(train_df, model_type='linear', debug=True)
    X_test,  y_test      = transform_preprocessor(test_df, tf)


    print("Processed shapes:", X_train.shape, X_test.shape)

    idata = fit_bayesian_hierarchical(
    df_clean,
    batter_idx=df_clean.batter_id.cat.codes.values,
    level_idx=df_clean.level_idx.values,
    age_centered=df_clean.age_centered.values,
    mu_prior=90,      # use league ave +/–5 mph
    sigma_prior=5
    )




