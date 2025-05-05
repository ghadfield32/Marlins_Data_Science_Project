
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from src.data.ColumnSchema import _ColumnSchema

# Optional imports with fallbacks for advanced statistics
try:
    import scipy.stats as stats  # type: ignore
    from statsmodels.nonparametric.smoothers_lowess import (
        lowess  # type: ignore
    )
    from statsmodels.stats.stattools import (
        durbin_watson  # type: ignore
    )
    _HAS_STATS_LIBS = True
except ImportError:
    _HAS_STATS_LIBS = False
    print("Warning: scipy or statsmodels not available. "
          "Some diagnostics will be limited.")

from scipy.stats import f_oneway, ttest_ind



def get_column_groups() -> dict:
    """
    Return a mapping of column-type → list of columns,
    based on the canonical schema in src.features.feature_selection.cols.
    """
    return cols.as_dict()

def check_nulls(df: pd.DataFrame):
    # Identify columns with null values
    null_columns = df.columns[df.isnull().any()].tolist()
    
    # Output the columns with null values
    if null_columns:
        print("Columns with null values:", null_columns)
    else:
        print("No columns with null values.")


def quick_pulse_check(
    df: pd.DataFrame,
    velo_col: str = "exit_velo",
    group_col: str = "batter_id",
    level_col: str = "level_abbr"
) -> pd.DataFrame:
    """
    Print a quick summary table:
      - total rows
      - unique batters
      - overall median exit_velo
      - median exit_velo by level
      - distribution of events per batter (median, 25th pct)
      - distribution of seasons per batter
      - pearson correlations of velo with launch_angle & hangtime
    Returns a pd.DataFrame with those metrics.
    """
    df = df.copy()
    total_rows = len(df)
    n_batters = df[group_col].nunique()
    overall_med = df[velo_col].median()

    # median by level
    med_by_level = df.groupby(level_col)[velo_col].median()

    # events per batter
    ev_per = df[group_col].value_counts()
    ev_stats = ev_per.quantile([0.25, 0.5]).to_dict()

    # seasons per batter
    seasons_per = df.groupby(group_col)["season"].nunique()
    seasons_stats = seasons_per.value_counts().sort_index().to_dict()

    # basic correlations
    corr = df[[velo_col, "launch_angle", "hangtime"]].corr()[velo_col].drop(velo_col)

    # Build a summary table
    metrics = [
        "Total rows",
        "Unique batters",
        "Overall median EV",
    ]
    values = [
        total_rows,
        n_batters,
        overall_med,
    ]
    
    # Add level-specific metrics
    for lvl in med_by_level.index:
        metrics.append(f"Median EV @ {lvl}")
        values.append(med_by_level[lvl])
    
    # Add batter event metrics
    metrics.extend([
        "Events per batter (25th pct)",
        "Events per batter (median)",
    ])
    values.extend([
        ev_stats.get(0.25, "N/A"),
        ev_stats.get(0.5, "N/A"),
    ])
    
    # Add season distribution
    for season_count, count in seasons_stats.items():
        metrics.append(f"Batters with {season_count} season(s)")
        values.append(count)
    
    # Add correlations
    metrics.extend([
        "ρ(exit_velo, launch_angle)",
        "ρ(exit_velo, hangtime)",
    ])
    values.extend([
        corr.get("launch_angle", "N/A"),
        corr.get("hangtime", "N/A"),
    ])

    table = pd.DataFrame({
        "Metric": metrics,
        "Value": values
    })
    
    print(table.to_string(index=False))
    return table


def red_flag_small_samples(df: pd.DataFrame,
                           group_col: str = "batter_id",
                           threshold: int = 15) -> pd.Series:
    """
    Identify batters with fewer than `threshold` events.
    Returns a Series of counts indexed by batter_id.
    """
    counts = df[group_col].value_counts()
    small = counts[counts < threshold]
    print(f"> Batters with fewer than {threshold} events: {len(small)}")
    if len(small) > 0:
        print(f"  First few: {', '.join(map(str, small.index[:5]))}")
    return small


def red_flag_level_effect(df: pd.DataFrame,
                          level_col: str = "level_abbr",
                          velo_col: str = "exit_velo") -> tuple:
    """
    One-way ANOVA of exit_velo across levels.
    Returns (F-statistic, p-value) or (None, None) if scipy is not available.
    """
    if not _HAS_STATS_LIBS:
        print("> ANOVA on exit_velo by level: scipy not available")
        print("> Basic level summary instead:")
        summary = df.groupby(level_col)[velo_col].agg(['mean', 'std', 'count'])
        print(summary)
        return None, None
    
    groups = [
        df[df[level_col] == lvl][velo_col].dropna()
        for lvl in df[level_col].unique()
    ]
    F, p = stats.f_oneway(*groups)
    print(f"> ANOVA on {velo_col} by {level_col}: F={F:.3f}, p={p:.3e}")
    return F, p


# ------------------------------------------------------------------
#  REPLACE old red_flag_level_effect  → clearer name & doc
# ------------------------------------------------------------------
def league_level_effect(
    df: pd.DataFrame,
    level_col: str = "level_abbr",
    velo_col: str = "exit_velo",
) -> tuple[float | None, float | None]:
    """
    🔹 Why it matters – confirms MLB vs Triple‑A (etc.) differences to
      justify hierarchical level effects in the model.

    One‑way ANOVA of `exit_velo` across `level_col`.
    Returns (F, p) or (None, None) if SciPy unavailable.
    """
    if not _HAS_STATS_LIBS:
        print("> SciPy unavailable – falling back to group summary")
        print(df.groupby(level_col)[velo_col].describe())
        return None, None

    groups = [df[df[level_col] == lv][velo_col].dropna()
              for lv in df[level_col].unique()]
    f_val, p_val = stats.f_oneway(*groups)
    print(f"> Level effect ANOVA: F={f_val:.3f}, p={p_val:.3e}")
    return f_val, p_val



def diag_age_effect(df: pd.DataFrame,
                    age_col: str = "age_centered",
                    velo_col: str = "exit_velo") -> np.ndarray | None:
    """
    LOWESS smoothing of exit_velo vs. age_centered.
    Returns the smoothed array or None if statsmodels is not available.
    """
    if not _HAS_STATS_LIBS:
        print("> Age effect analysis: statsmodels not available")
        print("> Basic correlation instead:")
        corr = df[[age_col, velo_col]].corr().iloc[0, 1]
        print(f"Correlation between {age_col} and {velo_col}: {corr:.3f}")
        return None
    
    # Run LOWESS smoothing
    smooth_result = lowess(df[velo_col], df[age_col])
    
    # Plot the result
    plt.figure(figsize=(6, 3))
    plt.scatter(df[age_col], df[velo_col], alpha=0.1, s=1, color='gray')
    plt.plot(
        smooth_result[:, 0], 
        smooth_result[:, 1], 
        'r-', 
        linewidth=2, 
        label="LOWESS fit"
    )
    plt.xlabel(age_col)
    plt.ylabel(velo_col)
    plt.title("Age effect (LOWESS)")
    plt.legend()
    plt.tight_layout()
    
    return smooth_result


def diag_time_series_dw(
    df: pd.DataFrame,
    time_col: str = "season",
    group_col: str = "batter_id",
    velo_col: str = "exit_velo"
) -> pd.Series | None:
    """
    Compute Durbin–Watson on each batter's time series of mean exit_velo.
    Returns a Series of DW statistics or None if statsmodels is not available.
    """
    if not _HAS_STATS_LIBS:
        print("> Time series analysis: statsmodels not available")
        return None
    
    # Create pivot table of seasons (columns) by batters (rows)
    pivot = (
        df
        .groupby([group_col, time_col])[velo_col]
        .mean()
        .unstack(fill_value=np.nan)
    )
    
    # Only process batters with at least 3 seasons
    valid_batters = pivot.dropna(thresh=3).index
    if len(valid_batters) == 0:
        print("> No batters with sufficient seasons for Durbin-Watson test")
        return None
    
    # Calculate DW statistic for each valid batter
    dw_stats = {}
    for batter in valid_batters:
        series = pivot.loc[batter].dropna()
        if len(series) >= 3:  # Recheck after dropna
            dw = durbin_watson(series)
            dw_stats[batter] = dw
    
    dw_series = pd.Series(dw_stats)
    print(
        f"> Mean Durbin–Watson across {len(dw_series)} batters: "
        f"{dw_series.mean():.3f}"
    )
    print("> DW < 1.5 suggests positive autocorrelation")
    print("> DW > 2.5 suggests negative autocorrelation")
    print("> DW ≈ 2.0 suggests no autocorrelation")
    
    return dw_series


# ------------------------------------------------------------------
#  REPLACE old diag_time_series_dw WITH optional helper
# ------------------------------------------------------------------
def _optional_dw_check(
    df: pd.DataFrame,
    time_col: str = "season",
    group_col: str = "batter_id",
    velo_col: str = "exit_velo",
) -> pd.Series | None:
    """
    (OPTIONAL) Durbin–Watson residual autocorrelation **per batter**.
    Mostly irrelevant for cross‑sectional EV analysis but retained
    behind a private name for power users.
    """
    if not _HAS_STATS_LIBS:
        return None
    pivot = (
        df.groupby([group_col, time_col])[velo_col]
          .mean().unstack()
    )
    stats_out = {}
    for idx, row in pivot.dropna(thresh=3).iterrows():
        if row.count() >= 3:
            stats_out[idx] = durbin_watson(row.dropna())
    if not stats_out:
        print("> DW check: no eligible batters")
        return None
    s = pd.Series(stats_out)
    print(f"DW mean={s.mean():.2f} (1.5<→pos autocorr, >2.5→neg)")
    return s




def check_red_flags(df: pd.DataFrame, 
                    sample_threshold: int = 15) -> dict:
    """
    Run all red flag checks and return the results in a dictionary.
    """
    results = {}
    
    # Check for small sample sizes
    small_samples = red_flag_small_samples(df, threshold=sample_threshold)
    results['small_samples'] = small_samples
    
    # Check for level effects
    f_stat, p_val = red_flag_level_effect(df)
    results['level_effect'] = {
        'f_statistic': f_stat,
        'p_value': p_val
    }
    
    return results


def plot_distributions(df: pd.DataFrame,
                       velo_col: str = "exit_velo",
                       by: str = "level_abbr"):
    """
    Histogram of `velo_col` faceted by `by`.
    Returns the Matplotlib figure so callers can save or show it.
    """
    groups = df[by].unique()
    fig, axes = plt.subplots(len(groups), 1,
                             figsize=(6, 2.8 * len(groups)),
                             sharex=True)
    for ax, grp in zip(axes, groups):
        ax.hist(df[df[by] == grp][velo_col], bins=30, alpha=0.75)
        ax.set_title(f"{by} = {grp} (n={len(df[df[by] == grp])})")
        ax.set_xlabel(velo_col)
    fig.tight_layout()
    return fig


def plot_correlations(df: pd.DataFrame, cols: list[str]):
    """
    Heat-map of Pearson correlations for `cols`.
    """
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(0.6 * len(cols) + 2,
                                    0.6 * len(cols) + 2))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(cols)), cols, rotation=90)
    ax.set_yticks(range(len(cols)), cols)
    fig.tight_layout()
    return fig


def plot_time_trends(df: pd.DataFrame,
                     time_col: str = "season",
                     group_col: str = "batter_id",
                     velo_col: str = "exit_velo",
                     sample: int = 50):
    """
    Plot mean exit-velo over time for a random sample of batters.
    """
    batters = df[group_col].unique()
    chosen = np.random.choice(batters,
                              min(sample, len(batters)),
                              replace=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for b in chosen:
        series = (
            df[df[group_col] == b]
            .groupby(time_col)[velo_col]
            .mean()
        )
        ax.plot(series.index, series.values, alpha=0.3)
    ax.set_xlabel(time_col)
    ax.set_ylabel(velo_col)
    ax.set_title("Sample batter exit-velo over time")
    fig.tight_layout()
    return fig


def summarize_numeric_vs_target(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    target_col: str = "exit_velo",
) -> pd.DataFrame:
    """
    Summarise each numeric predictor against the target.

    Returns a DataFrame indexed by feature with:
      n          – number of non‑null pairs
      pearson_r  – Pearson correlation coefficient
    """
    # --- Pull fresh lists from the schema every time -----------------
    groups = cols.as_dict()

    if numeric_cols is None:
        numeric_cols = groups.get("numerical", [])

    # --- Clean the list ---------------------------------------------
    numeric_cols = [
        c for c in numeric_cols
        if c != target_col and c in df.columns      # ❶ exclude target, ❷ guard
    ]

    records = []
    for col in numeric_cols:
        sub = df[[col, target_col]].dropna()
        if sub.empty:               # skip columns that are all‑NA
            continue
        r = sub[col].corr(sub[target_col])
        records.append({"feature": col, "n": len(sub), "pearson_r": r})

    result = (
        pd.DataFrame.from_records(records)
        .set_index("feature")
        .sort_values("pearson_r", ascending=False)
    )

    print("\n=== Numeric vs target correlations ===")
    print(result)

    return result


def plot_numeric_vs_target(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    target_col: str = "exit_velo",
):
    """
    Scatter plots of each numeric predictor vs the target with r‑value in title.
    """
    summary = summarize_numeric_vs_target(df, numeric_cols, target_col)
    for feature, row in summary.iterrows():
        plt.figure(figsize=(6, 4))
        plt.scatter(
            df[feature], df[target_col],
            alpha=0.3, s=5, edgecolors="none"
        )
        plt.title(f"{feature} vs {target_col}  (r = {row['pearson_r']:.2f})")
        plt.xlabel(feature)
        plt.ylabel(target_col)
        plt.tight_layout()
        plt.show()



def summarize_categorical_vs_target(
    df: pd.DataFrame,
    cat_cols: list[str] | None = None,
    target_col: str = "exit_velo"
) -> dict[str, pd.DataFrame]:
    """
    For each categorical feature, returns a DataFrame of:
      count, mean, median, std of the target by category.
    """
    groups = get_column_groups()
    if cat_cols is None:
        cat_cols = groups.get("categorical", [])

    summaries: dict[str, pd.DataFrame] = {}
    for col in cat_cols:
        stats = (
            df
            .groupby(col)[target_col]
            .agg(count="count", mean="mean", median="median", std="std")
            .sort_values("count", ascending=False)
        )
        print(f"\n=== {col} vs {target_col} summary ===")
        print(stats)
        summaries[col] = stats
    return summaries


def plot_categorical_vs_target(
    df: pd.DataFrame,
    cat_cols: list[str] | None = None,
    target_col: str = "exit_velo"
):
    """
    For each categorical feature, draw a box‑plot of the target by category.
    """
    groups = get_column_groups()
    if cat_cols is None:
        cat_cols = groups.get("categorical", [])

    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        df.boxplot(column=target_col, by=col, vert=False,
                   grid=False, patch_artist=True)
        plt.title(f"{target_col} by {col}")
        plt.suptitle("")           # remove pandas' automatic suptitle
        plt.xlabel(target_col)
        plt.tight_layout()
        plt.show()



def examine_and_filter_by_sample_size(
    df: pd.DataFrame,
    count_col: str = "exit_velo",
    group_col: str = "batter_id",
    season_col: str = "season",
    percentile: float = 0.05,
    min_count: int | None = None,
    filter_df: bool = False,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame | None]:
    """
    For each season:
      - compute per-batter count, mean, std of `count_col`
      - pick cutoff: min_count if provided, else the `percentile` quantile
      - print diagnostics
      - plot histograms *safely* (drops NaNs first)
    Returns:
      - summaries: dict season → per-batter summary DataFrame
      - filtered_df: if filter_df, the original df filtered to batters ≥ cutoff
    """
    summaries: dict[int, pd.DataFrame] = {}
    mask_keep: list[pd.Series] = []

    for season, sub in df.groupby(season_col):
        # 1) per-batter summary (count *non-NA* exit_velo)
        summary = (
            sub
            .groupby(group_col)[count_col]
            .agg(count="count", mean="mean", std="std")
            .sort_values("count")
        )
        summaries[season] = summary

        # 2) determine cutoff
        cutoff = min_count if min_count is not None else int(summary["count"].quantile(percentile))
        small = summary[summary["count"] < cutoff]
        large = summary[summary["count"] >= cutoff]

        # 3) diagnostics
        print(f"\n=== Season {season} (cutoff = {cutoff}) ===")
        print(f"  small (<{cutoff} events): {len(small)} batters")
        print(small[["count","mean","std"]].describe(), "\n")
        print(f"  large (≥{cutoff} events): {len(large)} batters")
        print(large[["count","mean","std"]].describe())

        # 4) **safe plotting**: drop NaNs, skip if nothing to plot
        small_means = small["mean"].dropna()
        large_means = large["mean"].dropna()

        if small_means.empty and large_means.empty:
            print(f"  ⚠️  Season {season}: no valid per-batter means to plot")
        else:
            plt.figure(figsize=(8, 3))
            if not small_means.empty:
                plt.hist(small_means, bins=30, alpha=0.6, label=f"n<{cutoff}")
            if not large_means.empty:
                plt.hist(large_means, bins=30, alpha=0.6, label=f"n≥{cutoff}")
            plt.title(f"Season {season}: per-batter EV means")
            plt.xlabel("Mean exit_velo")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # 5) build mask to keep only large-sample batters
        if filter_df:
            keep_ids = large.index
            mask_keep.append(
                (df[season_col] == season) &
                (df[group_col].isin(keep_ids))
            )

    # 6) combine masks and filter
    filtered_df = None
    if filter_df and mask_keep:
        combined = pd.concat(mask_keep, axis=1).any(axis=1)
        filtered_df = df[combined].copy()

    return summaries, filtered_df



def hypothesis_test(df, feature, target="exit_velo", test_type="anova"):
    """
    Perform hypothesis tests for feature significance.
    """
    if test_type == "anova":
        groups = [df[df[feature] == cat][target] for cat in df[feature].unique()]
        F, p = f_oneway(*groups)
        print(f"ANOVA: F={F:.3f}, p={p:.3e}")
        return F, p
    elif test_type == "ttest":
        group1 = df[df[feature] == 0][target]
        group2 = df[df[feature] == 1][target]
        t, p = ttest_ind(group1, group2)
        print(f"T-test: t={t:.3f}, p={p:.3e}")
        return t, p


# ------------------------------------------------------------------
#  NEW: robust outlier flagging
# ------------------------------------------------------------------
def flag_outliers_iqr(
    df: pd.DataFrame,
    velo_col: str = "exit_velo",
    iqr_mult: float = 1.5,
) -> pd.Series:
    """
    🔹 Why it matters – extreme EVs (>120 mph or <40 mph) can distort
      skew / variance estimates used in hierarchical priors.

    Returns a boolean Series (True = *suspect* outlier) using the
    classic IQR rule: value < Q1 − k·IQR  or  > Q3 + k·IQR.
    """
    q1, q3 = df[velo_col].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    mask = (df[velo_col] < lower) | (df[velo_col] > upper)
    n = int(mask.sum())
    print(f"> Outlier flag ({velo_col}): {n} rows outside [{lower:.1f}, {upper:.1f}]")
    return mask



# ------------------------------------------------------------------
#  NEW: EV distribution summary + QQ plot
# ------------------------------------------------------------------
def ev_distribution_summary(
    df: pd.DataFrame,
    velo_col: str = "exit_velo",
    bins: int = 40,
):
    """
    🔹 Why it matters – confirms right‑skew & heavy‑tail nature of EV
      so you can choose a skew‑normal or Student‑t likelihood.

    Prints skew/kurtosis, shows histogram, KDE, CDF & QQ (if scipy).
    """
    data = df[velo_col].dropna()
    print(
        f"Skewness = {stats.skew(data):.2f},  "
        f"Kurtosis = {stats.kurtosis(data, fisher=False):.2f}"
    )
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax[0].hist(data, bins=bins, density=True, alpha=0.7)
    data.plot(kind="kde", ax=ax[0], linewidth=2)
    ax[0].set_title("Histogram & KDE")

    # empirical CDF
    ecdf_x = np.sort(data)
    ecdf_y = np.arange(1, len(ecdf_x) + 1) / len(ecdf_x)
    ax[1].plot(ecdf_x, ecdf_y)
    ax[1].set_title("Empirical CDF")

    # QQ vs normal
    from scipy import stats as _st
    _st.probplot(data, dist="norm", plot=ax[2])
    ax[2].set_title("QQ‑plot vs Normal")
    plt.tight_layout()
    return fig


# ------------------------------------------------------------------
#  NEW: Year/era trend diagnostic
# ------------------------------------------------------------------
def year_trend_ev(
    df: pd.DataFrame,
    season_col: str = "season",
    velo_col: str = "exit_velo",
    ci: bool = True,
):
    """
    🔹 Why it matters – detects ball‑era shifts (e.g. 2019 “juiced”,
      2021 “deadened”) so forecasts for 2024 use correct baseline.

    Produces a table & line plot of mean/median EV per season.
    """
    g = df.groupby(season_col)[velo_col]
    stats_df = g.agg(mean="mean", median="median", n="count")
    print("\n=== Exit‑velo by season ===")
    print(stats_df)

    fig, ax = plt.subplots(figsize=(7, 3))
    stats_df["mean"].plot(ax=ax, marker="o", label="Mean EV")
    stats_df["median"].plot(ax=ax, marker="s", label="Median EV")
    if ci:
        sem = g.sem()
        ax.fill_between(
            stats_df.index,
            stats_df["mean"] - 1.96 * sem,
            stats_df["mean"] + 1.96 * sem,
            alpha=0.2,
            label="95% CI (mean)"
        )
    ax.set_ylabel(velo_col)
    ax.set_title("Seasonal trend in exit velocity")
    ax.legend()
    plt.tight_layout()
    return stats_df, fig



# ------------------------------------------------------------------
#  REPLACE old red_flag_level_effect  → clearer name & doc
# ------------------------------------------------------------------
def league_level_effect(
    df: pd.DataFrame,
    level_col: str = "level_abbr",
    velo_col: str = "exit_velo",
) -> tuple[float | None, float | None]:
    """
    🔹 Why it matters – confirms MLB vs Triple‑A (etc.) differences to
      justify hierarchical level effects in the model.

    One‑way ANOVA of `exit_velo` across `level_col`.
    Returns (F, p) or (None, None) if SciPy unavailable.
    """
    if not _HAS_STATS_LIBS:
        print("> SciPy unavailable – falling back to group summary")
        print(df.groupby(level_col)[velo_col].describe())
        return None, None

    groups = [df[df[level_col] == lv][velo_col].dropna()
              for lv in df[level_col].unique()]
    f_val, p_val = stats.f_oneway(*groups)
    print(f"> Level effect ANOVA: F={f_val:.3f}, p={p_val:.3e}")
    return f_val, p_val











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


    print("\n===== check on small samples =====")
    summaries, _ = examine_and_filter_by_sample_size(df, percentile=0.05)
    summaries, df_filtered = examine_and_filter_by_sample_size(
        df, percentile=0.05, min_count=15, filter_df=False
    )

    
    # Example usage
    print("\n===== NULLS CHECK =====")
    check_nulls(df_fe)
    
    print("\n===== QUICK PULSE CHECK =====")
    quick_pulse_check(df_fe)
    
    print("\n===== RED FLAGS CHECK =====")
    check_red_flags(df_fe)
    
    print("\n===== AGE EFFECT ANALYSIS =====")
    diag_age_effect(df_fe, age_col="age")
    
    print("\n===== TIME SERIES ANALYSIS =====")
    diag_time_series_dw(df_fe)
    
    print("\n===== PLOTTING =====")
    fig1 = plot_distributions(df_fe, by="hit_type")
    fig2 = plot_correlations(df_fe, numericals)  # Using cols schema
    fig3 = plot_time_trends(df_fe, sample=20)


    # — Numeric features —
    num_summary = summarize_numeric_vs_target(df_fe)
    plot_numeric_vs_target(df_fe)

    # — Categorical features —
    cat_summary = summarize_categorical_vs_target(df_fe)
    plot_categorical_vs_target(df_fe)

    # Example: Test if age has significant effect
    hypothesis_test(df_fe, feature="age_bin", test_type="anova")
    
    
    league_level_effect(df_fe)
    year_trend_ev(df_fe)
    flag_outliers_iqr(df_fe)
    ev_distribution_summary(df_fe)
# _optional_dw_check(df_fe)   # only if you still care
