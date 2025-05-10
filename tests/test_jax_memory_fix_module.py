import sys
import pathlib
import logging
import pytest
import jax
import jax.numpy as jnp

# Skip entire module if no GPU device is available for JAX
if not any(device.platform == 'gpu' for device in jax.devices()):
    pytest.skip("Skipping GPU tests since JAX GPU backend unavailable", allow_module_level=True)

from src.utils.jax_memory_fix_module import apply_jax_memory_fix, force_memory_allocation
from src.utils.jax_gpu_utils import (
    gpu_diagnostics,
    get_gpu_memory_info,
    check_jax_gpu_memory,
    force_gpu_memory_allocation,
    log_gpu_diagnostics
)

def test_apply_jax_memory_fix_returns_dict():
    settings = apply_jax_memory_fix(fraction=0.5, preallocate=False, verbose=False)
    assert isinstance(settings, dict), "Expected settings to be a dict"
    assert 'XLA_PYTHON_CLIENT_MEM_FRACTION' in settings, "Returned settings missing expected keys"


def test_force_memory_allocation_boolean():
    success = force_memory_allocation(size_mb=1)
    assert isinstance(success, bool), "force_memory_allocation should return a boolean"


def test_check_jax_gpu_memory_has_recommendations():
    mem_info = check_jax_gpu_memory()
    assert isinstance(mem_info, dict), "Expected memory info to be a dict"
    assert 'recommendations' in mem_info, "Missing 'recommendations' key in memory info"


def test_force_gpu_memory_allocation_boolean():
    success = force_gpu_memory_allocation(size_mb=1)
    assert isinstance(success, bool), "force_gpu_memory_allocation should return a boolean"


def test_get_gpu_memory_info_structure():
    info = get_gpu_memory_info()
    assert isinstance(info, dict), "GPU memory info should be a dict"
    # Depending on implementation, expect a list under 'nvidia_smi'
    assert 'nvidia_smi' in info, "Expected 'nvidia_smi' key in GPU memory info"
    assert isinstance(info['nvidia_smi'], list), "info['nvidia_smi'] should be a list"


def test_simple_jax_matmul():
    x = jnp.ones((10, 10))
    result = jnp.matmul(x, x).block_until_ready()
    assert result.shape == (10, 10), f"Unexpected result shape: {result.shape}"
    assert float(result.sum()) == pytest.approx(1000.0), "Sum of resulting matrix should be 1000.0"


def test_gpu_diagnostics_and_logging(caplog):
    caplog.set_level(logging.INFO)
    log_gpu_diagnostics()
    # At least one log record should be emitted
    assert len(caplog.records) > 0, "No log records captured from log_gpu_diagnostics"
