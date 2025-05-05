
"""
Gradient‑boosting baseline (XGBoost regressor).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
from sklearn.model_selection import cross_val_score

def _split_xy(df: pd.DataFrame):
    X = df.drop(columns=["exit_velo"])
    y = df["exit_velo"]
    return X, y


# At the top of src/models/gbm.py, after your imports:

from xgboost.core import XGBoostError

# ————— Detect GPU support —————
try:
    # Try a no-op instantiation to see if GPU build is present
    XGBRegressor(tree_method="gpu_hist", predictor="gpu_predictor")
    GPU_SUPPORT = True
    # Optional: print("✅  XGBoost GPU support detected")
except XGBoostError:
    GPU_SUPPORT = False
    # Optional: print("⚠️  XGBoost GPU support NOT available, falling back to CPU")

# ————— Updated tune_gbm —————
def tune_gbm(X, y, n_trials: int = 50):
    """
    Run an Optuna study to minimize CV RMSE of an XGBRegressor.
    Falls back to CPU if GPU is unavailable.
    """
    def objective(trial):
        # Base hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 0,
            "n_jobs": -1,
        }

        if GPU_SUPPORT:
            params.update({
                "tree_method": "gpu_hist",
                "predictor": "gpu_predictor",
            })
        else:
            params.update({
                "tree_method": "hist",
                "predictor": "cpu_predictor",
            })

        model = XGBRegressor(**params)
        # 3-fold CV, negative RMSE
        scores = cross_val_score(
            model, X, y,
            scoring="neg_root_mean_squared_error",
            cv=3, n_jobs=-1
        )
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_trial.params

# ————— Updated fit_gbm —————
def fit_gbm(X_tr, y_tr, X_te, y_te, **gbm_kw):
    """
    Train XGBRegressor with optional hyperparams, early stopping,
    and automatic GPU/CPU selection.
    Returns (model, RMSE).
    """
    # Default settings
    gbm_default = dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.7,
        colsample_bytree=0.6,
        random_state=0,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    # GPU vs CPU: only add the correct keys
    if GPU_SUPPORT:
        gbm_default.update({
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
        })
    else:
        gbm_default.update({
            "tree_method": "hist",
            "predictor": "cpu_predictor",
        })

    # Override with any user-passed kw
    gbm_default.update(gbm_kw)

    model = XGBRegressor(**gbm_default)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        early_stopping_rounds=gbm_default["early_stopping_rounds"],
        verbose=False
    )
    preds = model.predict(X_te)
    rmse = mean_squared_error(y_te, preds, squared=False)
    return model, rmse




if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_raw
    from src.features.feature_engineering import feature_engineer
    from src.data.ColumnSchema import _ColumnSchema
    from src.features.data_prep import clip_extreme_ev

    from sklearn.model_selection import train_test_split
    from src.features.preprocess import summarize_categorical_missingness
    from src.features.preprocess import fit_preprocessor, transform_preprocessor, inverse_transform_preprocessor
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

    # only on training data for linear/XGB
    train_df = clip_extreme_ev(train_df)
    #valid_df = clip_extreme_ev(valid_df)
    
    # run with debug prints
    X_train, y_train, tf = fit_preprocessor(train_df, model_type='linear', debug=True)
    X_test,  y_test      = transform_preprocessor(test_df, tf)

        
    print("Processed shapes:", X_train.shape, X_test.shape)

    # Example of inverse transform: 
    print("==========Example of inverse transform:==========")
    df_back = inverse_transform_preprocessor(X_train, tf)
    print("\n✅ Inverse‐transformed head (should mirror your original X_train):")
    print(df_back.head())
    print("Shape:", df_back.shape, "→ original X_train shape before transform:", X_train.shape)
    


    # === Hyperparameter tuning ===
    best_params = tune_gbm(X_train, y_train, n_trials=50)
    print("Tuned params:", best_params)

    # === Train & evaluate ===
    gbm_model, rmse = fit_gbm(
        X_train, y_train, X_test, y_test, **best_params
    )
    print(f"Tuned XGBoost RMSE: {rmse:.4f}")

