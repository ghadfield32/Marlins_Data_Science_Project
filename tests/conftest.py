
# tests/conftest.py
import pytest
import json
from src.utils.jax_memory_fix_module import apply_jax_memory_fix

@pytest.fixture(scope="session", autouse=True)
def _apply_jax_memory_fix():
    """
    Apply memory-fix **once** per test session before _anything_ imports JAX.
    Returns the settings dict so individual tests can assert on it.
    """
    settings = apply_jax_memory_fix(fraction=0.90, preallocate=True, verbose=False)
    yield settings  # let tests use it



