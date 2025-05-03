import pandas as pd

# ── NEW: model and importance imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import shap

from pathlib import Path
from src.data.load_data import load_raw
from src.features.feature_engineering import feature_engineer
from src.data.ColumnSchema import _ColumnSchema
# ── NEW: shapash and shapiq imports
from shapash import SmartExplainer
import shapiq
from sklearn.utils import resample

def train_baseline_model(X, y):
    """
    Fit a RandomForestRegressor on X, y.
    Returns the fitted model.
    """
    # You can adjust hyperparameters as needed
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model



def compute_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    n_jobs: int = 1,
    max_samples: float | int = None,
    random_state: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute permutation importances with controlled resource usage.

    Parameters
    ----------
    model : estimator
        Fitted model implementing .predict and .score.
    X : pd.DataFrame
        Features.
    y : pd.Series or array
        Target.
    n_repeats : int
        Number of shuffles per feature.
    n_jobs : int
        Number of parallel jobs (avoid -1 on Windows).
    max_samples : float or int, optional
        If float in (0,1], fraction of rows to sample.
        If int, absolute number of rows to sample.
    random_state : int
        Seed for reproducibility.
    verbose : bool
        Print debug info if True.

    Returns
    -------
    pd.DataFrame
        Columns: feature, importance_mean, importance_std.
        Sorted descending by importance_mean.
    """
    # Debug info
    if verbose:
        print(f"⏳ Computing permutation importances on {X.shape[0]} rows × {X.shape[1]} features")
        print(f"   n_repeats={n_repeats}, n_jobs={n_jobs}, max_samples={max_samples}")

    # Subsample if requested
    X_sel, y_sel = X, y
    if max_samples is not None:
        if isinstance(max_samples, float):
            nsamp = int(len(X) * max_samples)
        else:
            nsamp = int(max_samples)
        if verbose:
            print(f"   Subsampling to {nsamp} rows for speed")
        X_sel, y_sel = resample(X, y, replace=False, n_samples=nsamp, random_state=random_state)

    try:
        result = permutation_importance(
            model,
            X_sel, y_sel,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    except OSError as e:
        # Graceful fallback to single job
        if verbose:
            print(f"⚠️  OSError ({e}). Retrying with n_jobs=1")
        result = permutation_importance(
            model,
            X_sel, y_sel,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=1,
        )

    # Build and sort DataFrame
    importance_df = (
        pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    if verbose:
        print("✅ Permutation importances computed.")
    return importance_df


def compute_shap_importance(model, X, nsamples=100):
    """
    Compute mean absolute SHAP values per feature.
    Returns a DataFrame sorted by importance.
    """
    # DeepExplainer or TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    # sample for speed
    X_sample = X.sample(n=min(nsamples, len(X)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    # For regression, shap_values is a 2D array
    mean_abs_shap = pd.DataFrame({
        "feature": X_sample.columns,
        "shap_importance": np.abs(shap_values).mean(axis=0),
    })
    mean_abs_shap = mean_abs_shap.sort_values("shap_importance", ascending=False).reset_index(drop=True)
    return mean_abs_shap



def filter_permutation_features(
    importance_df: pd.DataFrame,
    threshold: float
) -> list[str]:
    """
    Return features whose permutation importance_mean exceeds threshold.
    """
    kept = importance_df.loc[
        importance_df["importance_mean"] > threshold, "feature"
    ]
    return kept.tolist()


def filter_shap_features(
    importance_df: pd.DataFrame,
    threshold: float
) -> list[str]:
    """
    Return features whose shap_importance exceeds threshold.
    """
    kept = importance_df.loc[
        importance_df["shap_importance"] > threshold, "feature"
    ]
    return kept.tolist()


def select_final_features(
    perm_feats: list[str],
    shap_feats: list[str],
    mode: str = "intersection"
) -> list[str]:
    """
    Combine permutation and SHAP feature lists.
    mode="intersection" for features in both lists,
    mode="union" for features in either list.
    """
    set_perm = set(perm_feats)
    set_shap = set(shap_feats)
    if mode == "union":
        final = set_perm | set_shap
    else:
        final = set_perm & set_shap
    # return sorted for reproducibility
    return sorted(final)



def load_final_features(
    file_path: str = "data/models/features/final_features.txt"
) -> list[str]:
    """
    Read the newline-delimited feature names file and return as a list.
    """
    with open(file_path, "r") as fp:
        return [line.strip() for line in fp if line.strip()]


def filter_to_final_features(
    df: pd.DataFrame,
    file_path: str = "data/models/features/final_features.txt"
) -> pd.DataFrame:
    """
    Given a feature-engineered DataFrame, load the final feature list,
    then return df[ ID_cols + final_features + [target] ].
    """
    # load the feature names
    final_feats = load_final_features(file_path)
    cols = _ColumnSchema()

    keep = cols.id() + final_feats + [cols.target()]
    missing = set(keep) - set(df.columns)
    if missing:
        raise ValueError(f"Cannot filter: missing columns {missing}")
    return df[keep].copy()





if __name__ == "__main__":
    # --- existing loading & schema logic ---
    raw_path = Path("data/Research Data Project/Research Data Project/exit_velo_project_data.csv")
    df = load_raw(raw_path)
    print(df.head())
    print(df.columns)
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if null_counts.empty:
        print("✅  No missing values in raw data.")
    else:
        print("=== Raw data null counts ===")
        for col, cnt in null_counts.items():
            print(f" • {col!r}: {cnt} missing")
    df_fe = feature_engineer(df)
    print("Raw →", df.shape, "//  Feature-engineered →", df_fe.shape)
    print(df_fe.head())

    cols = _ColumnSchema()
    print("ID columns:         ", cols.id())
    print("Ordinal columns:    ", cols.ordinal())
    print("Nominal columns:    ", cols.nominal())
    print("All categorical:    ", cols.categorical())
    print("Numerical columns:  ", cols.numerical())
    print("Model features:     ", cols.model_features())
    print("Target columns:     ", cols.target())
    print("All raw columns:    ", cols.all_raw())
    numericals = cols.numerical()
    numericals_without_y = [c for c in numericals if c not in cols.target()]

    # ── STEP 1: fully preprocess the engineered DataFrame ──
    from src.features.preprocess import fit_preprocessor, inverse_transform_preprocessor

    # fit_preprocessor returns (X_matrix, y, fitted_transformer)
    X_np, y, preproc = fit_preprocessor(df_fe, model_type='linear', debug=False)

    # Use the same index that y carries (only non-bunt, non-NA rows)
    idx = y.index

    # turn that into a DataFrame with the same column names:
    feat_names = preproc.get_feature_names_out()
    X = pd.DataFrame(X_np, columns=feat_names, index=idx)
    print(f"✅ Preprocessed feature matrix: {X.shape[0]} rows × {X.shape[1]} cols")

    # (optional) confirm inverse transform lines up:
    df_back = inverse_transform_preprocessor(X_np, preproc)
    df_back.index = idx
    print("✅ inverse_transform round-trip (head):")
    print(df_back.head())

    # ── STEP 2: train & compute importances on *that* X ──
    print("\nTraining baseline model…")
    model = train_baseline_model(X, y)

    print("\n🔍 Permutation Importances:")
    perm_imp = compute_permutation_importance(
        model, X, y,
        n_repeats=10,
        n_jobs=2,            # test small parallelism
        max_samples=0.5,     # test subsampling
        verbose=True
    )
    print(perm_imp)


    print("\n🔍 SHAP Importances:")
    shap_imp = compute_shap_importance(model, X)
    print(shap_imp)

    # ── STEP 3: threshold & select your final features ──
    perm_thresh = 0.01
    shap_thresh = 0.01
    perm_feats = filter_permutation_features(perm_imp, perm_thresh)
    shap_feats = filter_shap_features(shap_imp, shap_thresh)
    final_feats = select_final_features(perm_feats, shap_feats, mode="intersection")
    print(f"\nFinal preprocessed feature list ({len(final_feats)}):")
    print(final_feats)

    # ── STEP 4: build & save a dataset with just those features + target + IDs ──
    df_final = pd.concat([
        df_fe[cols.id()],
        df_fe[[cols.target()]],
        X[final_feats]
    ], axis=1)
    print("Final dataset shape:", df_final.shape)

    Path("data/models/features/final_features.txt").write_text("\n".join(final_feats))
    print("✔️ Saved feature list to final_features.txt")


    # Demo: filter the full df_fe back to just those features
    df_filtered = filter_to_final_features(df_fe)
    print("Filtered to final features shape:", df_filtered.shape)
