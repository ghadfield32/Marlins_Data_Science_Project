
"""
Gradient‑boosting baseline (XGBoost regressor).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping     # ① import the new callback
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
from src.data.ColumnSchema import _ColumnSchema
from xgboost.core import XGBoostError
from src.utils.gbm_utils import save_pipeline, load_pipeline

# ————— Detect GPU support using modern API —————
try:
    XGBRegressor(tree_method="hist", device="cuda")
    GPU_SUPPORT = True
except XGBoostError:
    GPU_SUPPORT = False

def _split_xy(df: pd.DataFrame):
    X = df.drop(columns=["exit_velo"])
    y = df["exit_velo"]
    return X, y

def tune_gbm(X, y, n_trials: int = 50):
    """
    Run an Optuna study to minimize CV RMSE of an XGBRegressor.
    Uses device=cuda if available, else CPU.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 0,
            "n_jobs": -1,
        }
        params.update({"tree_method": "hist"})
        model = XGBRegressor(**params)
        scores = cross_val_score(
            model, X, y,
            scoring="neg_root_mean_squared_error",
            cv=3, n_jobs=-1
        )
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params



def fit_gbm(X_tr, y_tr, X_te, y_te, **gbm_kw):
    """
    Train an XGBRegressor with optional hyperparameters and early stopping.
    Returns (fitted_model, rmse_on_test).
    """

    # A) Default constructor args
    constructor_defaults = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.6,
        random_state=0,
        n_jobs=-1,
        tree_method="hist",
    )
    constructor_defaults.update(gbm_kw)

    # B) Extract early_stopping_rounds (pop it out)
    early_stopping = constructor_defaults.pop("early_stopping_rounds", None)

    # C) Instantiate model
    model = XGBRegressor(**constructor_defaults)

    # D) Prepare fit kwargs
    fit_kwargs = {}
    # If an early_stopping value was provided, use the callback interface
    if early_stopping:
        fit_kwargs["callbacks"] = [EarlyStopping(rounds=early_stopping)]
        # pass eval_set so callback can work
        fit_kwargs["eval_set"] = [(X_te, y_te)]
    # You can still pass verbose if desired
    fit_kwargs["verbose"] = False

    # Fit the model with or without early stopping
    model.fit(X_tr, y_tr, **fit_kwargs)

    # E) Compute RMSE on test
    preds = model.predict(X_te)
    rmse  = mean_squared_error(y_te, preds, squared=False)

    return model, rmse





# ───────────────────────────────────────────────────────────────────────
# 6. Smoke test (only run when module executed directly)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_and_clean_data
    from src.features.feature_engineering import feature_engineer
    from src.features.preprocess import (fit_preprocessor
                                        ,transform_preprocessor
                                        ,inverse_transform_preprocessor)
    raw_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
    df = load_and_clean_data(raw_path)
    print(df.head())
    print(df.columns)
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


    train_df, test_df = train_test_split(df_fe, test_size=0.2, random_state=42)
    # run with debug prints
    # Record original feature count before preprocessing
    X_orig = train_df.drop(columns=["exit_velo"])
    orig_shape = X_orig.shape

    # Run preprocessing
    X_train, y_train, tf = fit_preprocessor(train_df, model_type='linear', debug=True)
    X_test,  y_test      = transform_preprocessor(test_df, tf)

    print("Processed shapes:", X_train.shape, X_test.shape)

    # Example of inverse transform: 
    print("==========Example of inverse transform:==========")
    df_back = inverse_transform_preprocessor(X_train, tf)
    print("\n✅ Inverse‐transformed head (should mirror your original X_train):")
    print(df_back.head())
    print(f"Shape: {df_back.shape} → original X_train shape before transform: {orig_shape}")




    # === Hyperparameter tuning ===
    best_params = tune_gbm(X_train, y_train, n_trials=5)
    print("Tuned params:", best_params)

    # === Train & evaluate ===
    gbm_model, rmse = fit_gbm(
        X_train, y_train, X_test, y_test, **best_params
    )
    print(f"Tuned XGBoost RMSE: {rmse:.4f}")


    # 3) save both at once
    save_pipeline(gbm_model, tf, path="models/gbm_pipeline.joblib")

    model, preprocessor = load_pipeline("models/gbm_pipeline.joblib")


