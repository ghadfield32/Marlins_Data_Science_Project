
# tests/test_hierarchical.py
import pytest
import numpy as np
import pandas as pd

# Import minimal pieces only after memory fix (fixture in conftest)
from src.models.hierarchical import fit_bayesian_hierarchical


def _synthetic_dataset(n=200):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "batter_id": rng.choice([f"B{i}" for i in range(8)], n),
        "pitcher_id": rng.choice([f"P{i}" for i in range(6)], n),
        "exit_velo": rng.normal(90, 4, n),
        "level_abbr": rng.choice(["A", "AA"], n),
        "season": rng.choice([2022, 2023], n),
        "age": rng.integers(18, 30, n),
    })
    # simple indices for test
    df["batter_idx"] = pd.Categorical(df.batter_id).codes
    df["level_idx"] = pd.Categorical(df.level_abbr).codes
    df["season_idx"] = pd.Categorical(df.season).codes
    df["pitcher_idx"] = pd.Categorical(df.pitcher_id).codes
    return df


@pytest.mark.gpu
@pytest.mark.slow
def test_hierarchical_smoke(tmp_path):
    """
    One end-to-end fit with tiny draws/tune to ensure model, memory monitor,
    and custom indices wire together.  Uses tmp_path so artifacts don’t
    clutter the repo.
    """
    df = _synthetic_dataset(200)
    idata = fit_bayesian_hierarchical(
        df,
        preprocessor=None,          # direct feature input in model wrapper
        batter_idx=df.batter_idx.to_numpy(),
        level_idx=df.level_idx.to_numpy(),
        season_idx=df.season_idx.to_numpy(),
        pitcher_idx=df.pitcher_idx.to_numpy(),
        sampler="jax",
        draws=20,
        tune=20,
        chains=1,
        monitor_memory=True,
        force_memory_allocation=False,
        allocation_target=0.5,
        direct_feature_input=None,  # model builds its own features
        out_dir=tmp_path            # assume your wrapper supports this kwarg
    )
    # very light assertion – just make sure sampling produced posterior
    assert "posterior" in idata
