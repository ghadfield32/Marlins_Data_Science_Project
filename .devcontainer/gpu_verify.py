#!/usr/bin/env python
"""
GPU Verification Script for Docker Container

This script runs at container startup to verify that:
1. JAX can access the GPU
2. Memory settings are properly applied
3. Basic operations work correctly
"""
import os
import sys
import time

# First ensure JAX memory settings are applied
try:
    # Add the project root to sys.path
    sys.path.insert(0, '/workspace')
    
    # Import our custom memory fix module
    from src.utils.jax_memory_fix_module import verify_gpu_usage, apply_jax_memory_fix
    
    # Apply memory settings - use settings from environment if available
    fraction = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9"))
    preallocate = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true").lower() == "true"
    
    print(f"\n{'='*80}\nGPU VERIFICATION STARTING\n{'='*80}")
    print(f"Applying JAX memory settings (fraction={fraction}, preallocate={preallocate})")
    
    # Apply the memory settings
    settings = apply_jax_memory_fix(
        fraction=fraction,
        preallocate=preallocate,
        verbose=True
    )
    
    # Import JAX after settings are applied
    import jax
    import jax.numpy as jnp
    
    # Print JAX configuration
    print(f"\nJAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX backend: {jax.lib.xla_bridge.get_backend().platform}")
    
    # Run the verification
    print("\nVerifying GPU usage with a test operation...")
    result = verify_gpu_usage(minimal_usage=0.1, operation_size=5000, verbose=True)
    
    if result.get("success", False):
        print(f"\n{'='*80}\nGPU VERIFICATION SUCCESS ✓\n{'='*80}")
        print(f"JAX is configured and using the GPU correctly")
        if "jax_reported_usage" in result:
            print(f"JAX reports {result['jax_reported_usage']:.2%} GPU memory used")
        sys.exit(0)
    else:
        print(f"\n{'='*80}\nGPU VERIFICATION FAILED ✗\n{'='*80}")
        print(f"Error: {result.get('error', 'Unknown error')}")
        # Don't exit with error to allow container to continue starting
        # sys.exit(1)
except Exception as e:
    print(f"\n{'='*80}\nGPU VERIFICATION ERROR ✗\n{'='*80}")
    print(f"Exception: {str(e)}")
    # Continue execution despite errors
    # sys.exit(1) 