#!/usr/bin/env python
"""
Quick memory test for JAX with direct memory allocations

This script tests the memory monitoring utilities with a simple JAX operation
to verify the memory allocation and monitoring are working correctly.
"""
import logging
import json
import numpy as np
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# First, import and apply memory settings (before importing JAX)
print("Importing and applying JAX memory settings...")
from src.utils.jax_memory_fix_module import apply_jax_memory_fix
settings = apply_jax_memory_fix(fraction=0.9, preallocate=True, verbose=True)

# Import memory monitoring
print("Importing memory monitor...")
from src.utils.jax_memory_monitor import (
    monitor_memory_usage,
    take_memory_snapshot,
    print_memory_snapshot,
    force_allocation_if_needed,
    generate_memory_report
)

# Now import JAX and GPU utils
print("Importing JAX...")
import jax
import jax.numpy as jnp
from src.utils.jax_gpu_utils import log_gpu_diagnostics

# Log GPU info
log_gpu_diagnostics()
print(f"Running on devices: {jax.devices()}")

def run_quick_test():
    """Run a quick test to verify memory allocation and monitoring."""
    print("\n=== Quick Memory Test ===\n")
    
    # Take initial snapshot
    initial = take_memory_snapshot("Initial state")
    print_memory_snapshot(initial)
    
    # Force memory allocation to target 70% utilization
    print("\nForcing memory allocation to 70%...")
    result = force_allocation_if_needed(
        target_fraction=0.7,
        current_usage_threshold=0.3,
        step_size_mb=2000,
        max_steps=6,
        verbose=True
    )
    print(f"Allocation result: {json.dumps(result, indent=2)}")
    
    # Take snapshot after allocation
    after_alloc = take_memory_snapshot("After allocation")
    print_memory_snapshot(after_alloc)
    
    # Run JAX operations with different matrix sizes
    sizes = [2000, 4000, 8000]
    for size in sizes:
        print(f"\nRunning matrix operation with size {size}x{size}...")
        
        # Create and multiply matrices
        with monitor_memory_usage(f"Matrix operation (size={size})"):
            x = jnp.ones((size, size), dtype=jnp.float32)
            y = jnp.ones((size, size), dtype=jnp.float32)
            z = jnp.matmul(x, y)
            z.block_until_ready()
    
    # Take final snapshot
    final = take_memory_snapshot("Final state")
    print_memory_snapshot(final)
    
    # Generate report
    report = generate_memory_report("quick_memory_report.json")
    
    # Print summary
    if "summary" in report and "gpu_utilization" in report["summary"]:
        util = report["summary"]["gpu_utilization"]
        print(f"\nTest Complete")
        print(f"GPU Utilization - Min: {util['min']:.2f}% Max: {util['max']:.2f}% Avg: {util['avg']:.2f}%")
    
    print("\nQuick memory test complete! Report saved to quick_memory_report.json")

if __name__ == "__main__":
    run_quick_test() 