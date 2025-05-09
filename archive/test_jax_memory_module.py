#!/usr/bin/env python
"""
Test script for JAX memory fixes and utilities

This script tests the functionality of our custom JAX memory fix module
and the GPU utilities to ensure they're working correctly.
"""
import os
import logging
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import and apply JAX memory fixes first (before JAX)
print("Importing and applying JAX memory fix...")
from src.utils.jax_memory_fix_module import apply_jax_memory_fix, force_memory_allocation

# Apply with custom settings
settings = apply_jax_memory_fix(fraction=0.9, preallocate=True, verbose=True)
print(f"Applied settings: {json.dumps(settings, indent=2)}")

# Now import JAX
print("\nImporting JAX and related modules...")
import jax
import jax.numpy as jnp

# Import GPU utilities
from src.utils.jax_gpu_utils import (
    gpu_diagnostics, 
    get_gpu_memory_info, 
    check_jax_gpu_memory,
    force_gpu_memory_allocation,
    log_gpu_diagnostics
)

# Log GPU diagnostics
print("\nLogging GPU diagnostics...")
log_gpu_diagnostics()

# Check JAX GPU memory
print("\nChecking JAX GPU memory configuration...")
memory_check = check_jax_gpu_memory()
print(f"Memory check recommendations: {json.dumps(memory_check['recommendations'], indent=2)}")

# Force memory allocation
print("\nForcing memory allocation...")
success = force_gpu_memory_allocation(size_mb=2000)
print(f"Memory allocation {'successful' if success else 'failed'}")

# Get memory info after allocation
print("\nGPU memory after allocation:")
memory_info = get_gpu_memory_info()
if memory_info and "nvidia_smi" in memory_info:
    for gpu in memory_info["nvidia_smi"]:
        print(f"GPU {gpu['gpu_id']}: {gpu['used_mb']} MB used / {gpu['total_mb']} MB total ({gpu['utilization']}%)")

# Run a simple JAX operation
print("\nRunning a simple JAX operation...")
x = jnp.ones((1000, 1000))
result = jnp.matmul(x, x).block_until_ready()
print(f"Matrix shape: {result.shape}, sum: {float(result.sum())}")

print("\nTest completed successfully!") 