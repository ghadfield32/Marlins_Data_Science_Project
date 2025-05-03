
"""
Frequentist mixed‑effects model using statsmodels MixedLM.

Formula implemented:
    exit_velo ~ 1 + level_ord + age_centered
              + (1 | batter_id)

We rely on columns already produced by preprocess():
    • level_idx  (0,1,2)   – ordinal
    • age_centered
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_mixed(train: pd.DataFrame,
              test: pd.DataFrame):
    """Return (fitted model, RMSE on test)."""
    # statsmodels wants a *single* DataFrame with all cols
    # so we concatenates and keep row positions for slicing
    combined = pd.concat([train, test], axis=0)
    # ensure categorical dtype
    combined["level_ord"] = combined["level_idx"].astype(int)

    mdl = smf.mixedlm(
        formula="exit_velo ~ 1 + level_ord + age_centered",
        data=combined.iloc[: len(train)],
        groups=combined.iloc[: len(train)]["batter_id"]
    ).fit(reml=False)

    # predict on test set
    pred = mdl.predict(exog=combined.iloc[len(train):])
    true = test["exit_velo"].values
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    return mdl, rmse


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
