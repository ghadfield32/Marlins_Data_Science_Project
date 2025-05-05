
"""
Fast linear baselines (OLS and Ridge).

Usage
-----
>>> from src.models.linear import fit_ridge
>>> fitted, rmse = fit_ridge(train_df, val_df)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


def _split_xy(df: pd.DataFrame):
    X = df.drop(columns=["exit_velo"])
    y = df["exit_velo"]
    return X, y


def fit_ridge(X_tr: pd.DataFrame,
              y_tr: pd.DataFrame,
              X_te: pd.DataFrame,
              y_te: pd.DataFrame,
              alpha: float = 1.0):
    """
    Returns (sklearn Pipeline, RMSE on test set).
    """

    model = Pipeline(
        [("reg" , Ridge(alpha=alpha, random_state=0))]
    ).fit(X_tr, y_tr)

    pred = model.predict(X_te)
    rmse = np.sqrt(np.mean((pred - y_te) ** 2))
    return model, rmse



if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_raw
    from src.features.feature_engineering import feature_engineer
    from src.data.ColumnSchema import _ColumnSchema
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
    

    # === NEW: Train & evaluate Ridge regression ===
    model_ridge, rmse_ridge = fit_ridge(X_train, y_train, X_test,  y_test)
    print(f"Ridge regression RMSE: {rmse_ridge:.4f}")
