import arviz as az
import pandas as pd
import numpy as np
from src.utils.bayesian_explainability import (
    plot_parameter_forest, posterior_table, plot_ppc, shap_explain
)
from src.data.load_data import load_raw        # for loading raw CSV data
from src.features.feature_engineering import feature_engineer
from src.features.preprocess import prepare_for_mixed_and_hierarchical




def predict(df: pd.DataFrame, idata) -> np.ndarray:
    """
    Generate predictions using the posterior mean of the hierarchical model.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared DataFrame with 'level_idx' and 'age_centered' columns.
    idata : arviz.InferenceData
        Fitted model inference data to base predictions on.

    Returns
    -------
    np.ndarray
        Array of predicted exit velocities for each row in df.
    """
    posterior_mean = idata.posterior.mean(dim=("chain", "draw"))
    coef_level = posterior_mean["beta_level"].values
    coef_age = posterior_mean["beta_age"].values
    mu_hat = posterior_mean["mu"].values
    return mu_hat + coef_level[df["level_idx"].values] + coef_age * df["age_centered"].values


if __name__ == "__main__":
    from src.models.hierarchical_utils import save_model, load_model
    # === Editable settings ===
    # Path to the saved model (NetCDF format)
    MODEL_PATH = "data/models/saved_models/exitvelo_hmc.nc"
    # Input data for prediction (raw CSV with exit velocity data)
    raw_path = "data/Research Data Project/Research Data Project/exit_velo_validate_data.csv"
    # Output predictions file (CSV) or set to None to print to console
    OUTPUT_PREDS_2024 = "data/predictions/predictions_2024.csv"  # <-- EDITABLE: set output CSV path or None
    # Season filter for prediction/explanation
    SEASON = 2024  
    # Notes:
    # - To retrain the model (train mode), you would import and call `fit_bayesian_hierarchical` from src.models.hierarchical,
    #   passing MU_PRIOR, SIGMA_PRIOR, SAMPLER, DRAWS, and TUNE settings as needed.
    # - Example training settings (not used in predict mode):
    #     MU_PRIOR = 90       # Prior mean for global intercept
    #     SIGMA_PRIOR = 5     # Scale for variability parameters
    #     SAMPLER = "jax"    # Options: "jax", "nutpie", "advi"
    #     DRAWS, TUNE = 1000, 1000

    # ---------------- Data loading & preparation ----------------
    df_raw = load_raw(raw_path)           # load raw CSV into DataFrame
    df_fe = feature_engineer(df_raw)         # apply feature engineering
    df_model = prepare_for_mixed_and_hierarchical(df_fe)

    #save_model(idata, "data/models/saved_models/exitvelo_hmc.nc")
    load_model(MODEL_PATH)
    # extract index arrays for model features (required if retraining)
    batter_idx = df_model["batter_id"].cat.codes.values  # categorical codes
    level_idx = df_model["level_idx"].values
    age_centered = df_model["age_centered"].values

    # --------------------- Load Saved Model ---------------------
    idata = load_model(MODEL_PATH)

    # --------------------- Generate Predictions ---------------------
    # Filter to the season of interest for explanation
    df_season = df_model[df_model["season"] == SEASON].copy()
    if df_season.empty:
        raise ValueError(f"No data found for season {SEASON}")
    
    preds_season = predict(df_season, idata)
    df_season["predicted_exit_velo"] = preds_season

    # ------------------ Output Predictions 2024 ------------------
    if OUTPUT_PREDS_2024:
        # Write only the key columns to CSV in one go
        df_season[[
            "batter_id", "level_idx", "age_centered",
            "exit_velo", "predicted_exit_velo"
        ]].to_csv(OUTPUT_PREDS_2024, index=False)
        print(f"2024 season predictions written to {OUTPUT_PREDS_2024}")

    # ------------------ Summary Statistics ------------------
    print(f"\n=== Season {SEASON} Exit Velocity Summary ===")
    actual_desc = df_season["exit_velo"].describe()
    pred_desc = pd.Series(preds_season, name="predicted").describe()
    print("Actual exit_velo stats:\n", actual_desc)
    print("\nPredicted exit_velo stats:\n", pred_desc)

    # ------------- Model Parameter Summaries -------------
    print("\n=== Model Parameter Credibility ===")
    param_table = posterior_table(idata)
    print(param_table)
    plot_parameter_forest(idata, var_names=["mu", "beta_level", "beta_age", "sigma_b"])


    # ----------- Posterior Predictive Check -----------
    print("\n=== Posterior Predictive Check ===")
    plot_ppc(idata)

    # ---------------- Feature Importances ----------------
    print("\n=== SHAP Feature Importances ===")
    # Capture the returned SHAP values array
    shap_values = shap_explain(
        predict_fn=lambda df: predict(df, idata),
        background_df=df_season.sample(min(200, len(df_season)), random_state=0),
        sample_df=df_season.sample(min(200, len(df_season)), random_state=1),
    )

    print("\nTop-5 SHAP feature indices (highest mean absolute impact):")
    top5 = np.abs(shap_values).mean(0).argsort()[-5:][::-1]
    print(top5)
