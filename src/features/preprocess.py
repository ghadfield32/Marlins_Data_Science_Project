"""
Preprocessing module for exit velocity pipeline.
Supports multiple model types (linear, XGBoost, PyMC, etc.) with
automatic ordinal-category detection from the data.
"""
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data.ColumnSchema import _ColumnSchema
from sklearn.model_selection import train_test_split

from pandas.api.types import is_categorical_dtype
# ───────────────────────────────────────────────────────────────────────
# Numeric & nominal pipelines (unchanged)
# ───────────────────────────────────────────────────────────────────────
numeric_linear = Pipeline([
    ('impute', SimpleImputer(strategy='median', add_indicator=True)),
    ('scale', StandardScaler()),
])
numeric_iterative = Pipeline([
    ('impute', IterativeImputer(random_state=0, add_indicator=True)),
    ('scale', StandardScaler()),
])
nominal_pipe = Pipeline([
    ('impute', SimpleImputer(strategy='constant', fill_value='MISSING')),
    ('encode', OneHotEncoder(drop='first', handle_unknown='ignore')),
])

# ───────────────────────────────────────────────────────────────────────
# Dynamic preprocess functions
# ───────────────────────────────────────────────────────────────────────
# src/features/preprocess.py

def fit_preprocessor(
    df: pd.DataFrame,
    model_type: str = "linear",
    debug: bool = False,
) -> tuple[np.ndarray, pd.Series, ColumnTransformer]:
    """
    Build & fit the preprocessing ColumnTransformer on the *full* training data.
    Returns (X_matrix, y, fitted_transformer).
    """
    cols = _ColumnSchema()
    TARGET = cols.target()

    # ------------------------------------------------------------
    # 1. filter rows & coerce numerics
    # ------------------------------------------------------------
    df = df[df["hit_type"].str.upper() != "BUNT"].copy()
    df = df.dropna(subset=[TARGET])
    num_feats = [c for c in cols.numerical() if c != TARGET]
    df[num_feats] = df[num_feats].apply(pd.to_numeric, errors="coerce")

    # engineered auxiliaries
    df["age_centered"] = df["age"] - df["age"].median()
    df["level_idx"] = df["level_abbr"].map({"AA": 0, "AAA": 1, "MLB": 2})

    # ------------------------------------------------------------
    # 2. Prepare X, y as DATAFRAMES (keeps column names)
    # ------------------------------------------------------------
    ord_feats = cols.ordinal()
    nom_feats = cols.nominal()
    X = df[num_feats + ord_feats + nom_feats]
    y = df[TARGET]

    # force all ordinal columns to string so categories are comparable
    X[ord_feats] = (
        X[ord_feats]
        .astype(str)
        .where(X[ord_feats].notna(), other=np.nan)  # keep NaNs
    )

    # ------------------------------------------------------------
    # 3. Compute *global* ordinal category lists
    # ------------------------------------------------------------
    ordinal_categories = []
    for c in ord_feats:
        cats = (
            X[c].dropna().unique().tolist()
        )
        if "MISSING" not in cats:
            cats.append("MISSING")
        ordinal_categories.append(cats)

    if debug:
        print("Detected ordinal categories:", list(zip(ord_feats, ordinal_categories)))

    # ------------------------------------------------------------
    # 4. Build pipelines
    # ------------------------------------------------------------
    ordinal_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            (
                "encode",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype="int32",
                ),
            ),
        ]
    )

    numeric_pipe = (
        numeric_linear if model_type == "linear" else numeric_iterative
    )

    ct = ColumnTransformer(
        [
            ("num", numeric_pipe, num_feats),
            ("ord", ordinal_pipe, ord_feats),
            ("nom", nominal_pipe, nom_feats),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_mat = ct.fit_transform(X, y)  # still returns a NumPy array
    return X_mat, y, ct




def transform_preprocessor(
    df: pd.DataFrame,
    transformer: ColumnTransformer,
) -> tuple[np.ndarray, pd.Series]:
    """
    Apply an already‑fitted transformer to *any* new DataFrame.
    Unseen ordinal categories are coerced to 'MISSING' first.
    """
    cols = _ColumnSchema()
    TARGET = cols.target()
    num_feats = [c for c in cols.numerical() if c != TARGET]
    ord_feats = cols.ordinal()
    nom_feats = cols.nominal()

    df = df.dropna(subset=[TARGET]).copy()
    df[num_feats] = df[num_feats].apply(pd.to_numeric, errors="coerce")
    df["age_centered"] = df["age"] - df["age"].median()
    df["level_idx"] = df["level_abbr"].map({"AA": 0, "AAA": 1, "MLB": 2})

    X = df[num_feats + ord_feats + nom_feats]
    y = df[TARGET]

    # unseen ordinals → 'MISSING'
    X[ord_feats] = (
        X[ord_feats]
        .astype(str)
        .where(X[ord_feats].notna(), other="MISSING")
    )

    X_mat = transformer.transform(X)  # no warnings now
    return X_mat, y



def inverse_transform_preprocessor(
    X_trans: np.ndarray,
    transformer: ColumnTransformer
) -> pd.DataFrame:
    """
    Invert each block of a ColumnTransformer back to its original features,
    based on the exact column lists we passed in.
    """
    import numpy as np, pandas as pd

    # 1) Flatten the lists we gave each transformer to recover original feature order
    orig_features: list[str] = []
    for name, _, cols in transformer.transformers_:
        if cols == 'drop':
            continue
        orig_features.extend(cols)

    parts = []
    start = 0
    n_rows = X_trans.shape[0]

    # 2) For each transformer, slice & inverse-transform
    for name, trans, cols in transformer.transformers_:
        if cols == 'drop':
            continue

        fitted = transformer.named_transformers_[name]

        # how many columns did it produce?
        dummy = np.zeros((1, len(cols)))
        try:
            out = fitted.transform(dummy)
        except Exception:
            out = dummy
        n_out = out.shape[1]

        block = X_trans[:, start:start + n_out]
        start += n_out

        # apply inverse_transform
        if trans == 'passthrough':
            inv = block
        elif name == 'num':
            scaler = fitted.named_steps['scale']
            inv_full = scaler.inverse_transform(block)
            inv = inv_full[:, :len(cols)]
        else:
            if isinstance(fitted, Pipeline):
                last = list(fitted.named_steps.values())[-1]
                inv = last.inverse_transform(block)
            else:
                inv = fitted.inverse_transform(block)

        parts.append(pd.DataFrame(inv, columns=cols, index=range(n_rows)))

    # 3) Concatenate & reorder
    df_orig = pd.concat(parts, axis=1)
    return df_orig[orig_features]

def prepare_for_mixed_and_hierarchical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a feature-engineered df, drop bunts & missing target,
    add age_centered, level_idx, and make batter_id categorical.
    """
    cols = _ColumnSchema()
    TARGET = cols.target()

    df = df.copy()
    # 1) Drop bunts & missing target
    df = df[df["hit_type"].str.upper() != "BUNT"]
    df = df.dropna(subset=[TARGET])

    # 2) Center age & index levels
    df["age_centered"] = df["age"] - df["age"].median()
    df["level_idx"]   = df["level_abbr"].map({"AA": 0, "AAA": 1, "MLB": 2})

    # 3) Categorical batter_id
    df["batter_id"] = df["batter_id"].astype("category")

    return df




# debugs:
def summarize_categorical_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each categorical column (ordinal + nominal), compute:
      - original_null_count / pct
      - imputed_missing_count / pct
    Safely handles pandas.Categorical by first adding 'MISSING' to its categories.
    """
    cols    = _ColumnSchema()
    cat_cols = cols.ordinal() + cols.nominal()
    summary = []
    n = len(df)

    for col in cat_cols:
        ser = df[col]
        orig_null = ser.isna().sum()

        # If it's a Categorical, add 'MISSING' as a valid category
        if is_categorical_dtype(ser):
            ser = ser.cat.add_categories(['MISSING'])

        # Count rows that would become 'MISSING'
        imputed_missing = ser.fillna('MISSING').eq('MISSING').sum()

        summary.append({
            'column': col,
            'original_null_count':   orig_null,
            'original_null_pct':     orig_null / n,
            'imputed_missing_count': imputed_missing,
            'imputed_missing_pct':   imputed_missing / n,
        })

    return pd.DataFrame(summary)





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

    # Example of inverse transform: 
    print("==========Example of inverse transform:==========")
    df_back = inverse_transform_preprocessor(X_train, tf)
    print("\n✅ Inverse‐transformed head (should mirror your original X_train):")
    print(df_back.head())
    print("Shape:", df_back.shape, "→ original X_train shape before transform:", X_train.shape)

