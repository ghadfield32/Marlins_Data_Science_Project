""" 
Column schema helper for exit‑velo project.

Centralises every raw and engineered column name in one place and exposes
 type‑safe accessors so downstream code never hard‑codes strings.

Usage
-----
>>> from src.features.columns import cols
>>> num_cols = cols.numerical()
>>> ord_cols = cols.ordinal()
>>> all_for_model = cols.model_features()
"""

from functools import lru_cache
from typing import List, Dict
import json

class _ColumnSchema:
    """Container for canonical column lists.

    Keeping everything behind methods avoids accidental mutation and lets
    IDEs offer autocompletion (because the return type is always `List[str]`).
    """

    _ID_COLS: List[str] = [
        "season", "batter_id", "pitcher_id",
    ]

    _ORDINAL_CAT_COLS: List[str] = [
        "level_abbr",   # AA < AAA < MLB
        "age_bin",      # 4 quantile bins of age
        "la_bin",       # 5 quantile bins of launch angle
        "spray_bin",    # 3 quantile bins of spray angle
        # "outcome",      # categorical outcome (out, single, double, triple, home run, sacrifice fly, sacrifice bunt, fielders choice)
    ]


    _NOMINAL_CAT_COLS = [
        "hit_type",
        "pitch_group",
        "batter_hand",
        "pitcher_hand",
        "hand_match",
        "pitch_hand_match",
        "same_hand",
        "hitter_type", 
    ]

    _NUMERICAL_COLS = [
        "launch_angle",
        "spray_angle",
        "hangtime",
        "height_diff",
        "age_sq",
        "age_centered",
        "season_centered",
        "level_idx",
        "player_ev_mean50",
        "player_ev_std50",
        "pitcher_ev_mean50",
        # "power_rate",
        # "season_power_80", 
    ]


    _TARGET_COL: str = "exit_velo"

    # ────────────────────────────────────────────────────────────────────
    # Public helpers
    # ────────────────────────────────────────────────────────────────────
    def id(self) -> List[str]:
        return self._ID_COLS.copy()

    def ordinal(self) -> List[str]:
        return self._ORDINAL_CAT_COLS.copy()

    def nominal(self) -> List[str]:
        return self._NOMINAL_CAT_COLS.copy()

    def categorical(self) -> List[str]:
        """All cat cols (ordinal + nominal)."""
        return self._ORDINAL_CAT_COLS + self._NOMINAL_CAT_COLS

    def numerical(self) -> List[str]:
        return self._NUMERICAL_COLS.copy()


    def target(self) -> str:
        return self._TARGET_COL

    # ------------------------------------------------------------------
    @lru_cache(maxsize=1)
    def model_features(self) -> List[str]:
        """Columns fed into the ML pipeline *after* preprocess.

        Excludes the target but includes derived cols.
        """
        phys_minus_target = [
            c for c in self._NUMERICAL_COLS if c != self._TARGET_COL
        ]

        return (
            phys_minus_target
            + self._NOMINAL_CAT_COLS
            + self._ORDINAL_CAT_COLS  # some algos want raw string order
        )

    def all_raw(self) -> List[str]:
        """Returns every column expected in raw input CSV (incl. engineered)."""
        return (
            self._ID_COLS
            + self._ORDINAL_CAT_COLS
            + self._NOMINAL_CAT_COLS
            + self._NUMERICAL_COLS
        )

    def as_dict(self) -> Dict[str, List[str]]:
        """Dictionary form – handy for YAML/JSON dumps."""
        return {
            "id": self.id(),
            "ordinal": self.ordinal(),
            "nominal": self.nominal(),
            "numerical": self.numerical(),
            "target": [self.target()],
        }


if __name__ == "__main__":
    cols = _ColumnSchema()

    __all__ = ["cols"]
    print("ID columns:         ", cols.id())
    print("Ordinal columns:    ", cols.ordinal())
    print("Nominal columns:    ", cols.nominal())
    print("All categorical:    ", cols.categorical())
    print("Numerical columns:  ", cols.numerical())
    print("Target column:      ", cols.target())
    print("Model features:     ", cols.model_features())
    print("All raw columns:    ", cols.all_raw())
    print("\nAs dict (JSON):")
    print(json.dumps(cols.as_dict(), indent=2))

