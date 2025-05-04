from src.utils.bayesian_explainability import (
    plot_parameter_forest, posterior_table, plot_ppc, shap_explain
)


if __name__ ==
    # --- NEW: helper for SHAP ---------------------------------
    posterior_mean = idata.posterior.mean(dim=("chain", "draw"))
    coef_level = posterior_mean["beta_level"].values
    coef_age = posterior_mean["beta_age"].values
    mu_hat = posterior_mean["mu"].values

    def predict(df):
        return (
            mu_hat
            + coef_level[df["level_idx"].values]
            + coef_age * df["age_centered"].values
        )

    pmjax._numpyro_stats_to_dict.__globals__["np"].log2 = _debug_log2

    # 1️⃣  Convergence / coefficient credibility
    plot_parameter_forest(idata, var_names=["beta_level", "beta_age"])

    # After fitting Bayesian model:
    summary = posterior_table(idata)
    print("Posterior Table with Significance:\n", summary)
    # 2️⃣  Posterior‑predictive overlay
    plot_ppc(idata)

    # 3️⃣  WAIC & LOO (single‐line outputs)
    print("WAIC:", az.waic(idata).waic, "  LOO:", az.loo(idata).loo)

    # 4️⃣  Global & local feature importance
    shap_explain(
        predict_fn=predict,
        background_df=df_model.sample(200, random_state=0),
        sample_df=df_model.sample(200, random_state=1),
    )

