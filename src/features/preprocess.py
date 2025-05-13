"""
Preprocessing module for exit velocity pipeline.
Supports multiple model types (linear, XGBoost, PyMC, etc.) with
automatic ordinal-category detection from the data.
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, IterativeImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data.ColumnSchema import _ColumnSchema
from sklearn.model_selection import train_test_split
from src.features.data_prep import (filter_bunts_and_popups
                                    , compute_clip_bounds
                                    , clip_extreme_ev
                                    , filter_low_event_batters
                                    , filter_physical_implausibles)


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
# Helper function for domain cleaning to reduce duplication
# ───────────────────────────────────────────────────────────────────────
def filter_and_clip(
    df: pd.DataFrame,
    lower: float = None,
    upper: float = None,
    quantiles: tuple[float, float] = (0.01, 0.99),
    debug: bool = False
) -> pd.DataFrame:
    """
    Clean the dataset by:
    1. Filtering out bunts and popups
    2. Clipping extreme exit velocity values
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to clean
    lower : float, optional
        Lower bound for clipping, computed from data if None
    upper : float, optional
        Upper bound for clipping, computed from data if None
    quantiles : tuple[float, float], optional
        Quantiles to use if computing bounds from data
    debug : bool, optional
        Whether to print diagnostic information
        
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (copy of input)
    tuple[float, float]
        The (lower, upper) bounds used for clipping
    """
    cols = _ColumnSchema()
    TARGET = cols.target()
    
    # 1. Filter out bunts and popups
    df = filter_bunts_and_popups(df, debug=debug)
    
    # 2. Compute bounds if not provided
    if lower is None or upper is None:
        lower_computed, upper_computed = compute_clip_bounds(
            df[TARGET], method="quantile", quantiles=quantiles
        )
        lower = lower if lower is not None else lower_computed
        upper = upper if upper is not None else upper_computed
    
    # 3. Clip extreme values
    df = clip_extreme_ev(df, lower=lower, upper=upper, debug=debug)

    # 1) drop batters with too few events
    df = filter_low_event_batters(df, batter_col="batter_id", min_events=15, debug=debug)

    # 2) drop physical implausibles
    df = filter_physical_implausibles(
        df,
        hang_col="hangtime",
        velo_col="exit_velo",
        debug=debug)
    
    return df, (lower, upper)

# ───────────────────────────────────────────────────────────────────────
# Dynamic preprocess functions
# ───────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 1) fit_preprocessor (with train‐set‐only bound computation + clip)  
# ─────────────────────────────────────────────────────────────────────────────
def fit_preprocessor(
    df: pd.DataFrame,
    model_type: str = "linear",
    debug: bool = False,
    quantiles: tuple[float, float] = (0.01, 0.99),
    max_safe_rows: int = 200000
) -> tuple[np.ndarray, pd.Series, ColumnTransformer]:
    """
    Fit a preprocessing pipeline to transform raw data into model-ready features,
    adding a MissingIndicator for ordinal features.

    IMPORTANT: Call this on TRAINING data only.
    """
    if len(df) > max_safe_rows:
        warnings.warn(
            f"Dataset has {len(df)} rows (>{max_safe_rows}); "
            "ensure you’re only fitting on TRAINING data.",
            UserWarning
        )

    # 0) Domain cleaning and clipping
    df, (lower, upper) = filter_and_clip(df, quantiles=quantiles, debug=debug)

    cols = _ColumnSchema()
    TARGET = cols.target()

    # 1) Split out features + coerce numerics
    num_feats = [c for c in cols.numerical() if c != TARGET]
    ord_feats = cols.ordinal()
    nom_feats = cols.nominal()

    df[num_feats] = df[num_feats].apply(pd.to_numeric, errors="coerce")
    X = df[num_feats + ord_feats + nom_feats].copy()
    y = df[TARGET]

    # 2) Prepare ordinal columns
    X[ord_feats] = X[ord_feats].astype("string")
    X.loc[:, ord_feats] = X.loc[:, ord_feats].mask(X[ord_feats].isna(), np.nan)
    ordinal_categories = [[*X[c].dropna().unique(), "MISSING"] for c in ord_feats]
    ordinal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("encode", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype="int32"
        )),
    ])

    # 3) Numeric pipeline selection
    if model_type == "linear":
        num_imputer = SimpleImputer(strategy="median", add_indicator=True)
    else:
        num_imputer = IterativeImputer(
            random_state=0,
            add_indicator=True
        )

    numeric_pipe = Pipeline([
        ("impute", num_imputer),
        ("scale", StandardScaler()),
    ])

    # 4) ColumnTransformer with an extra MissingIndicator for ordinals
    ct = ColumnTransformer(
        [
            ("num", numeric_pipe, num_feats),
            ("ord_ind", MissingIndicator(missing_values=np.nan), ord_feats),
            ("ord", ordinal_pipe, ord_feats),
            ("nom", nominal_pipe, nom_feats),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # preserve clipping bounds for transform
    ct.lower_, ct.upper_ = lower, upper

    X_mat = ct.fit_transform(X, y)
    return X_mat, y, ct




# ─────────────────────────────────────────────────────────────────────────────
def transform_preprocessor(
    df: pd.DataFrame,
    transformer: ColumnTransformer,
) -> tuple[np.ndarray, pd.Series]:
    """
    Transform new data using a fitted preprocessor.
    
    Parameters
    ----------
    df : pd.DataFrame
        New data to transform
    transformer : ColumnTransformer
        Fitted transformer from fit_preprocessor
        
    Returns
    -------
    X_mat : np.ndarray
        Transformed feature matrix
    y : pd.Series
        Target values
    """
    cols, TARGET = _ColumnSchema(), _ColumnSchema().target()

    # Domain filter & clip using the bounds from the fitted transformer
    df, _ = filter_and_clip(df, lower=transformer.lower_, upper=transformer.upper_)

    # rebuild lists & coerce numerics
    num_feats = [c for c in cols.numerical() if c != TARGET]
    ord_feats = cols.ordinal()
    nom_feats = cols.nominal()
    df[num_feats] = df[num_feats].apply(pd.to_numeric, errors="coerce")

    X = df[num_feats + ord_feats + nom_feats].copy()

    # **NEW** – ensure ordinals are strings, then fill unseen→"MISSING"
    X[ord_feats] = X[ord_feats].astype("string")
    X.loc[:, ord_feats] = X.loc[:, ord_feats].where(X.loc[:, ord_feats].notna(), "MISSING")

    y = df[TARGET]
    X_mat = transformer.transform(X)
    return X_mat, y


def inverse_transform_preprocessor(
    X_trans: np.ndarray,
    transformer: ColumnTransformer
) -> pd.DataFrame:
    """
    Invert each block of a ColumnTransformer back to its original features,
    skipping transformers that lack inverse_transform (e.g., MissingIndicator).
    """
    import numpy as np
    import pandas as pd
    import warnings
    from sklearn.impute import MissingIndicator  # for isinstance check

    # 1) Recover original feature order
    orig_features: list[str] = []
    for name, _, cols in transformer.transformers_:
        if cols == 'drop': continue
        orig_features.extend(cols)

    parts = []
    start = 0
    n_rows = X_trans.shape[0]

    # 2) Loop through each transformer block
    for name, trans, cols in transformer.transformers_:
        if cols == 'drop':
            continue

        fitted = transformer.named_transformers_[name]

        # 2a) Determine number of output columns
        dummy = np.zeros((1, len(cols)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                out = fitted.transform(dummy)
            except Exception:
                out = dummy
        n_out = out.shape[1]

        # 2b) Slice the real transformed data block
        block = X_trans[:, start : start + n_out]
        start += n_out

        # 2c) Handle each transformer type
        if isinstance(fitted, MissingIndicator):
            # Skip MissingIndicator: it has no inverse_transform
            continue  # :contentReference[oaicite:2]{index=2}
        elif trans == 'passthrough':
            inv = block
        elif name == 'num':
            scaler = fitted.named_steps['scale']
            inv_full = scaler.inverse_transform(block)
            inv = inv_full[:, :len(cols)]
        else:
            # Ordinal or nominal pipelines
            if isinstance(fitted, Pipeline):
                last = list(fitted.named_steps.values())[-1]
                inv = last.inverse_transform(block)
            else:
                inv = fitted.inverse_transform(block)

        parts.append(pd.DataFrame(inv, columns=cols, index=range(n_rows)))

    # 3) Concatenate & reorder to match original features
    df_orig = pd.concat(parts, axis=1)
    return df_orig[orig_features]





def prepare_for_mixed_and_hierarchical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the rows *and* adds convenience covariates expected by the
    hierarchical and mixed-effects models.
    """
    cols = _ColumnSchema()
    TARGET = cols.target()

    # Apply domain cleaning with default quantiles
    df, _ = filter_and_clip(df.copy())

    # Category coding for batter
    df["batter_id"] = df["batter_id"].astype("category")

    # New: Category coding for season
    df["season_cat"] = df["season"].astype("category")
    df["season_idx"] = df["season_cat"].cat.codes

    # New: Category coding for pitcher
    df["pitcher_cat"] = df["pitcher_id"].astype("category")
    df["pitcher_idx"] = df["pitcher_cat"].cat.codes

    return df





def prepare_for_pymc(df: pd.DataFrame,
                     predictor: str,
                     target: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    1) Domain-clean with filter_and_clip
    2) Drop any rows where predictor or target is NaN
    3) Return (clean_df, x_array, y_array)
    """
    # 1) domain cleaning + clipping
    df_clean, _bounds = filter_and_clip(df.copy(), debug=False)

    # 2) drop rows with missing predictor or target
    df_clean = df_clean.dropna(subset=[predictor, target]).reset_index(drop=True)

    # 3) extract numpy arrays
    x = df_clean[predictor].values
    y = df_clean[target].values

    return df_clean, x, y





# ───────────────────────────────────────────────────────────────────────
# 6. Smoke test (only run when module executed directly)
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_and_clean_data
    from src.features.feature_engineering import feature_engineer
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
    X_train, y_train, tf = fit_preprocessor(train_df, model_type='linear', debug=True)
    X_test,  y_test      = transform_preprocessor(test_df, tf)

    print("Processed shapes:", X_train.shape, X_test.shape)

    # Example of inverse transform: 
    print("==========Example of inverse transform:==========")
    df_back = inverse_transform_preprocessor(X_train, tf)
    print("\n✅ Inverse‐transformed head (should mirror your original X_train):")
    print(df_back.head())
    print("Shape:", df_back.shape, "→ original X_train shape before transform:", X_train.shape)



