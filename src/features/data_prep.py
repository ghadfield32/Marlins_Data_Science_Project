import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
def compute_clip_bounds(
    series: pd.Series,
    *,
    method: str = "quantile",
    quantiles: tuple[float,float] = (0.01,0.99),
    std_multiplier: float = 3.0,
) -> tuple[float,float]:
    """
    Compute (lower, upper) but do not apply them.
    """
    s = series.dropna()
    if method == "quantile":
        return tuple(s.quantile(list(quantiles)).to_list())
    if method == "mean_std":
        mu, sigma = s.mean(), s.std()
        return mu - std_multiplier*sigma, mu + std_multiplier*sigma
    if method == "iqr":
        q1, q3 = s.quantile([0.25,0.75])
        iqr = q3 - q1
        return q1 - 1.5*iqr, q3 + 1.5*iqr
    raise ValueError(f"Unknown method {method}")


def clip_extreme_ev(
    df: pd.DataFrame,
    velo_col: str = "exit_velo",
    lower: float | None = None,
    upper: float | None = None,
    *,
    method: str = "quantile",
    quantiles: tuple[float, float] = (0.01, 0.99),
    std_multiplier: float = 3.0,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Clip exit velocities to [lower, upper].

    If lower/upper are None, compute them from the data using one of:
      - method="quantile": use df[velo_col].quantile(quantiles)
      - method="mean_std":  use mean ± std_multiplier * std
      - method="iqr":       use [Q1 - 1.5*IQR, Q3 + 1.5*IQR]

    When debug=True, prints counts & percentages of values that will be clipped.
    """
    df = df.copy()
    series = df[velo_col].dropna()

    # 1) infer bounds if not given
    if lower is None or upper is None:
        if method == "quantile":
            low_q, high_q = quantiles
            lower_, upper_ = series.quantile([low_q, high_q]).to_list()
        elif method == "mean_std":
            mu, sigma = series.mean(), series.std()
            lower_, upper_ = mu - std_multiplier * sigma, mu + std_multiplier * sigma
        elif method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_, upper_ = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        else:
            raise ValueError(f"Unknown method '{method}' for clip_extreme_ev")
        lower  = lower  if lower  is not None else lower_
        upper  = upper  if upper  is not None else upper_

    # 2) debug: count how many will be clipped
    if debug:
        total   = len(series)
        n_low   = (series < lower).sum()
        n_high  = (series > upper).sum()
        print(f"[clip_extreme_ev] lower={lower:.2f}, upper={upper:.2f}")
        print(f"  → {n_low:,} / {total:,} ({n_low/total:.2%}) below lower")
        print(f"  → {n_high:,} / {total:,} ({n_high/total:.2%}) above upper")

    # 3) actually clip
    df[velo_col] = df[velo_col].clip(lower, upper)
    return df


# ─────────────────────────────────────────────────────────────────────────────
def filter_bunts_and_popups(
    df: pd.DataFrame,
    hit_col: str = "hit_type",
    debug: bool = False
) -> pd.DataFrame:
    """
    Drop bunts and pop-ups from the DataFrame.
    If debug=True, prints how many rows were removed.
    """
    df = df.copy()
    initial = len(df)
    mask = ~df[hit_col].str.upper().isin(["BUNT", "POP_UP"])
    filtered = df[mask]
    if debug:
        removed = initial - len(filtered)
        print(f"[filter_bunts_and_popups] dropped {removed:,} rows "
              f"({removed/initial:.2%}) bunts/pop-ups")
    return filtered


def filter_low_event_batters(
    df: pd.DataFrame,
    batter_col: str = "batter_id",
    min_events: int = 15,
    debug: bool = False
) -> pd.DataFrame:
    """
    Drop all rows for batters with fewer than min_events total events.
    """
    df = df.copy()
    counts = df[batter_col].value_counts()
    keep_batters = counts[counts >= min_events].index
    initial = len(df)
    filtered = df[df[batter_col].isin(keep_batters)]
    if debug:
        removed = initial - len(filtered)
        print(f"[filter_low_event_batters] dropped {removed:,} rows ({removed/initial:.2%}) for batters with <{min_events} events")
    return filtered


def filter_physical_implausibles(
    df: pd.DataFrame,
    hang_col: str = "hangtime",
    velo_col: str = "exit_velo",
    min_hang: float = 0.5,
    max_velo: float = 115.0,
    debug: bool = False
) -> pd.DataFrame:
    """
    Drop rows where hangtime < min_hang AND exit_velo > max_velo,
    as these are likely sensor glitches (e.g., foul tips).
    """
    df = df.copy()
    initial = len(df)
    mask = ~((df[hang_col] < min_hang) & (df[velo_col] > max_velo))
    filtered = df[mask]
    if debug:
        removed = initial - len(filtered)
        print(f"[filter_physical_implausibles] dropped {removed:,} rows ({removed/initial:.2%}) for hangtime<{min_hang} & exit_velo>{max_velo}")
    return filtered




# ─────────────────────────────────────────────────────────────────────────────
# Smoke test / CLI entry
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    from src.data.load_data import load_and_clean_data
    from src.data.ColumnSchema import _ColumnSchema
    from src.features.eda import summarize_categorical_missingness
    from src.features.feature_engineering import feature_engineer
    # from src.features.data_prep import filter_and_clip
    raw_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
    df = load_and_clean_data(raw_path)
    print(df.head())
    print(df.columns)

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


    df_fe = feature_engineer(df)

    summary_df = summarize_categorical_missingness(df_fe)
    print(summary_df.to_markdown(index=False))

    print("Raw →", df.shape, "//  Feature‑engineered →", df_fe.shape)
    print(df_fe.head())

    # filter out bunts and popups
    df_fe = filter_bunts_and_popups(df_fe)
    # chekc on bunts and popups
    print(df_fe["hit_type"].unique())
    print(df_fe["outcome"].unique())

    debug = True
    TARGET = cols.target()
    lower, upper = compute_clip_bounds(
        df[TARGET],
        method="quantile",            # default: 1st/99th percentile
        quantiles=(0.01, 0.99),
    )
    if debug:
        total = len(df)
        n_low = (df[TARGET] < lower).sum()
        n_high= (df[TARGET] > upper).sum()
        print(f"[fit_preprocessor] clipping train EV to [{lower:.2f}, {upper:.2f}]")
        print(f"  → {n_low:,}/{total:,} ({n_low/total:.2%}) below")
        print(f"  → {n_high:,}/{total:,} ({n_high/total:.2%}) above")
    df_clipped = clip_extreme_ev(df, lower=lower, upper=upper)

    print("Final rows after filter & clip:", len(df_clipped))


    # 1) drop batters with too few events
    df_fe = filter_low_event_batters(df_fe, batter_col="batter_id", min_events=15, debug=debug)

    # 2) drop physical implausibles
    df_fe = filter_physical_implausibles(
        df_fe,
        hang_col="hangtime",
        velo_col="exit_velo",
        debug=debug)
