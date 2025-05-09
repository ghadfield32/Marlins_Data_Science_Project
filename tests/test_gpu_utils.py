
# tests/test_gpu_utils.py
import json
import pytest
from src.utils.jax_gpu_utils import (
    log_gpu_diagnostics,
    get_gpu_memory_info,
    check_jax_gpu_memory,
)

def test_gpu_diagnostics_smoke(caplog):
    """Just make sure the function runs without exception and logs something."""
    log_gpu_diagnostics()
    assert caplog.records, "No log records produced by log_gpu_diagnostics"


def test_memory_info_structure():
    """`get_gpu_memory_info()` should return a dict (or None on CPU)."""
    info = get_gpu_memory_info()
    if info is None:            # CPU-only CI lanes
        pytest.skip("No GPU available – skipping GPU memory info check")
    assert "nvidia_smi" in info or "jax" in info


def test_recommendations_keys():
    recs = check_jax_gpu_memory()
    expect = {"status", "recommendations"}
    assert expect.issubset(recs), f"Missing keys in {json.dumps(recs)}"
