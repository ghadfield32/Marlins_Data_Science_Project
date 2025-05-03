# src/features/eda.py

import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
from src.features.feature_selection import cols

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


def get_column_groups() -> dict:
    """
    Return a mapping of column-type → list of columns,
    based on the canonical schema in src.features.feature_selection.cols.
    """
    return cols.as_dict()


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


if __name__ == "__main__":
    from src.data.load_data import load_raw
    path = 'data/Research Data Project/Research Data Project/exit_velo_project_data.csv'
    df_train = load_raw(path)

    print("\n===== QUICK PULSE CHECK =====")
    quick_pulse_check(df_train)

    print("\n===== RED FLAGS CHECK =====")
    check_red_flags(df_train)

    print("\n===== AGE EFFECT ANALYSIS =====")
    diag_age_effect(df_train, age_col="age")

    print("\n===== TIME SERIES ANALYSIS =====")
    diag_time_series_dw(df_train)

    print("\n===== PLOTTING =====")
    fig1 = plot_distributions(df_train, by="hit_type")
    fig2 = plot_correlations(df_train, cols.numerical())  # Using cols schema
    fig3 = plot_time_trends(df_train, sample=20)


