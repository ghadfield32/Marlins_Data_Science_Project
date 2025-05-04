import arviz as az
import shap, numpy as np
import matplotlib.pyplot as plt

# ---------------- Posterior summaries -----------------
def plot_parameter_forest(idata, var_names=None, hdi_prob=0.95):
    """Caterpillar/forest plot of posterior estimates."""
    return az.plot_forest(
        idata,
        var_names=var_names,
        combined=True,
        hdi_prob=hdi_prob,
        kind="forest",
        figsize=(6, len(var_names or idata.posterior.data_vars) * 0.4),
    )

def posterior_table(idata, round_to=2):
    """
    Return a nicely rounded HDI/mean table with significance.
    """
    summary = az.summary(idata, hdi_prob=0.95).round(round_to)
    summary["significant"] = (summary["hdi_2.5%"] > 0) | (summary["hdi_97.5%"] < 0)
    return summary

# ---------------- Posterior‑predictive checks ---------
def plot_ppc(idata, kind="overlay"):
    """Visual PPC (over‑laid densities by default)."""
    return az.plot_ppc(idata, kind=kind, alpha=0.1)

# ---------------- SHAP-based feature importances ------
def shap_explain(predict_fn, background_df, sample_df):
    """
    Model‑agnostic Kernel SHAP on the *posterior mean predictor*.

    predict_fn(df) must return a 1‑D numpy array of predictions.
    """
    explainer = shap.KernelExplainer(predict_fn, background_df)
    shap_values = explainer.shap_values(sample_df, nsamples=200)
    shap.summary_plot(shap_values, sample_df, show=False)
    plt.tight_layout()
    return shap_values
