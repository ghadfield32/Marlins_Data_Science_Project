# tests/conftest.py
import pytest
import json
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.jax_memory_fix_module import apply_jax_memory_fix

# Register custom markers
def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: marks tests that require a GPU")
    config.addinivalue_line("markers", "slow: marks tests that are slow to run")

@pytest.fixture(scope="session", autouse=True)
def _apply_jax_memory_fix():
    """
    Apply memory-fix **once** per test session before _anything_ imports JAX.
    Returns the settings dict so individual tests can assert on it.
    """
    settings = apply_jax_memory_fix(fraction=0.90, preallocate=True, verbose=False)
    yield settings  # let tests use it



