"""
Preprocessing module for exit velocity pipeline.
Supports multiple model types (linear, XGBoost, PyMC, etc.) with
automatic ordinal-category detection from the data.
"""
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
import warnings

from src.data.ColumnSchema import _ColumnSchema
from src.features.data_prep import clip_extreme_ev

# pipelines unchanged
numeric_linear = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])
numeric_iterative = Pipeline([
    ("impute", IterativeImputer(random_state=0, add_indicator=True)),
    ("scale", StandardScaler()),
])
nominal_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("encode", OneHotEncoder(drop="first", handle_unknown="ignore")),
])

def fit_preprocessor(df: pd.DataFrame, model_type: str = "linear", debug: bool = False):
    """
    Fit ColumnTransformer on full training data.
    Returns (X_matrix, y, fitted_transformer).
    """
    cols = _ColumnSchema()
    TARGET = cols.target()

    # 1) Filter & coerce
    df = df[df["hit_type"].str.upper() != "BUNT"].copy()
    df = df.dropna(subset=[TARGET])
    num_feats = [c for c in cols.numerical() if c != TARGET]
    df[num_feats] = df[num_feats].apply(pd.to_numeric, errors="coerce")

    # 2) Assemble
    ord_feats = cols.ordinal()
    nom_feats = cols.nominal()
    X = df[num_feats + ord_feats + nom_feats].copy()
    y = df[TARGET]

    if debug:
        # summary of ordinal columns
        dtypes = dict(X[ord_feats].dtypes)
        uniques = {c: X[c].nunique() for c in ord_feats}
        print(f"DEBUG: Ordinals ({len(ord_feats)}): dtypes={dtypes}, uniques={uniques}")
        print("DEBUG: Sample rows:")
        print(X[ord_feats].head(2))

    # convert ordinal to string dtype
    for c in ord_feats:
        X.loc[:, c] = [str(v) if pd.notna(v) else np.nan for v in X[c].values]

    if debug:
        uniques_after = {c: X[c].nunique() for c in ord_feats}
        print(f"DEBUG: After conversion uniques={uniques_after}")

    # 3) Build category lists
    ordinal_categories = []
    for c in ord_feats:
        cats = sorted([v for v in pd.unique(X[c]) if pd.notna(v)], key=lambda x: str(x))
        if "MISSING" not in cats:
            cats.append("MISSING")
        ordinal_categories.append(cats)
        X.loc[X[c].isna(), c] = "MISSING"

    if debug:
        sizes = {ord_feats[i]: len(ordinal_categories[i]) for i in range(len(ord_feats))}
        print(f"DEBUG: Category sizes={sizes}")

    # 4) Build transformer
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("encode", OrdinalEncoder(categories=ordinal_categories,
                                   handle_unknown="use_encoded_value",
                                   unknown_value=-1, dtype="int32")),
    ])
    numeric_pipe = numeric_linear if model_type == "linear" else numeric_iterative
    ct = ColumnTransformer([
        ("num", numeric_pipe, num_feats),
        ("ord", ordinal_pipe, ord_feats),
        ("nom", nominal_pipe, nom_feats),
    ], remainder="drop", verbose_feature_names_out=False)

    # 5) Fit & transform
    X_mat = ct.fit_transform(X, y)

    # stash
    ct._ord_feats = ord_feats
    ct._ord_cats  = ordinal_categories

    if debug:
        print(f"DEBUG: Transformer fitted -> X_mat shape={X_mat.shape}")
        try:
            names = ct.get_feature_names_out()
            print(f"DEBUG: Features count={len(names)}")
        except Exception:
            pass

    return X_mat, y, ct


def transform_preprocessor(df: pd.DataFrame, transformer: ColumnTransformer, debug: bool = False):
    """
    Apply fitted transformer. Returns (X_matrix, y).
    """
    cols = _ColumnSchema()
    TARGET = cols.target()
    df = df[df["hit_type"].str.upper() != "BUNT"].copy()
    df = df.dropna(subset=[TARGET])
    num_feats = [c for c in cols.numerical() if c != TARGET]
    df[num_feats] = df[num_feats].apply(pd.to_numeric, errors="coerce")

    ord_feats = transformer._ord_feats
    nom_feats = cols.nominal()
    X = df[num_feats + ord_feats + nom_feats].copy()
    y = df[TARGET]

    if debug:
        uniques = {c: X[c].nunique() for c in ord_feats}
        print(f"DEBUG: Pre-conversion uniques={uniques}")

    # convert ordinal to string
    for c in ord_feats:
        X.loc[:, c] = [str(v) if pd.notna(v) else np.nan for v in X[c].values]

    # remap unseen
    unseen = {}
    for i, c in enumerate(ord_feats):
        vals = set(X[c].dropna()) - set(transformer._ord_cats[i])
        if vals:
            unseen[c] = len(vals)
            X.loc[~X[c].isin(transformer._ord_cats[i]), c] = "MISSING"
    if debug:
        print(f"DEBUG: Unseen values={unseen or 'None'}")

    X_mat = transformer.transform(X)
    if debug:
        print(f"DEBUG: Transformed shape={X_mat.shape}")
    return X_mat, y


def inverse_transform_preprocessor(X_trans: np.ndarray,
                                   transformer: ColumnTransformer,
                                   debug: bool = False) -> pd.DataFrame:
    """
    Invert each block of a ColumnTransformer back to its original features.
    If debug=True, prints a summary per-block.
    """
    orig_feats = [
        col
        for name, _, cols in transformer.transformers_
        if cols != 'drop'
        for col in cols
    ]
    parts = []
    start = 0
    n_rows = X_trans.shape[0]

    for name, trans, cols in transformer.transformers_:
        if cols == 'drop':
            continue

        fitted = transformer.named_transformers_[name]

        # figure out how many columns it produced
        dummy = np.zeros((1, len(cols)))
        try:
            out = fitted.transform(dummy)
        except Exception:
            out = dummy
        n_out = out.shape[1]

        # slice the block
        end = start + n_out
        block = X_trans[:, start:end]
        start = end

        if debug:
            print(f"→ Block '{name}': input_cols={len(cols)}, output_cols={n_out}, block_shape={block.shape}")

        # inverse-transform if possible
        if name == 'num' and hasattr(fitted, 'named_steps'):
            block = fitted.named_steps['scale'].inverse_transform(block)
        elif trans != 'passthrough' and hasattr(fitted, 'inverse_transform'):
            block = fitted.inverse_transform(block)

        # create DataFrame, with fallback if shapes disagree
        if n_out == len(cols):
            df_block = pd.DataFrame(block, columns=cols, index=range(n_rows))
        else:
            if debug:
                print(f"⚠️  Shape mismatch in '{name}': "
                      f"{n_out} outputs vs {len(cols)} input names—using generic names")
            generic = [f"{name}_{i}" for i in range(n_out)]
            df_block = pd.DataFrame(block, columns=generic, index=range(n_rows))

        parts.append(df_block)

    df_orig = pd.concat(parts, axis=1)
    if debug:
        print(f"🔄 Final concatenated shape: {df_orig.shape}  (expected columns ≤ {len(orig_feats)})")

    existing = [c for c in orig_feats if c in df_orig.columns]
    return df_orig[existing]



def prepare_for_mixed_and_hierarchical(df: pd.DataFrame) -> pd.DataFrame:
    cols = _ColumnSchema()
    TARGET = cols.target()
    df = df.copy()
    df = df[df["hit_type"].str.upper() != "BUNT"]
    df = df.dropna(subset=[TARGET])
    df["batter_id"] = df["batter_id"].astype("category")
    return df


def summarize_categorical_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each categorical column (ordinal + nominal), compute:
      - original_null_count / pct
      - imputed_missing_count / pct
    Safely handles pandas.Categorical by adding 'MISSING' before fillna.
    Drops rows with missing target to focus analysis on valid outcomes.
    """
    cols = _ColumnSchema()
    TARGET = cols.target()
    # Remove rows missing target
    df_valid = df.dropna(subset=[TARGET])
    cat_cols = cols.ordinal() + cols.nominal()
    summary = []
    n = len(df_valid)

    for col in cat_cols:
        ser = df_valid[col]
        orig_null = ser.isna().sum()
        # If categorical, add 'MISSING' category
        if is_categorical_dtype(ser):
            ser = ser.cat.add_categories('MISSING')
        # Count imputed missing including existing 'MISSING'
        imp_missing = ser.fillna('MISSING').eq('MISSING').sum()
        summary.append({
            'column': col,
            'orig_null': orig_null,
            'orig_pct': orig_null / n,
            'imp_null': imp_missing,
            'imp_pct': imp_missing / n,
        })
    return pd.DataFrame(summary)



# ───────────────────────────────────────────────────────────────────────
# 6. Smoke test (only run when module executed directly)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_raw
    from src.features.feature_engineering import feature_engineer
    from src.features.data_prep import clip_extreme_ev
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
    

