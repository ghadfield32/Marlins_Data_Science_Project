#!/usr/bin/env python
"""
Test script for JAX memory monitoring during training

This script tests the memory monitoring utilities to verify they can
track GPU memory usage during model training operations.
"""
import os
import logging
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import memory fix and apply (must come before importing JAX)
print("Importing and applying JAX memory fix...")
from src.utils.jax_memory_fix_module import apply_jax_memory_fix

# Apply memory settings
settings = apply_jax_memory_fix(fraction=0.9, preallocate=True, verbose=True)
print(f"Applied settings: {json.dumps(settings, indent=2)}")

# Now import memory monitor and JAX
print("\nImporting memory monitor and JAX...")
from src.utils.jax_memory_monitor import (
    monitor_memory_usage,
    take_memory_snapshot,
    print_memory_snapshot,
    force_allocation_if_needed,
    generate_memory_report
)
import jax
import jax.numpy as jnp

# Import GPU utilities
from src.utils.jax_gpu_utils import log_gpu_diagnostics

print("\nRunning on devices:", jax.devices())
log_gpu_diagnostics()

def simulate_training_step(size=5000, compute_time=2.0):
    """Simulate a training step with matrix operations."""
    print(f"\nSimulating training step (size={size}, compute_time={compute_time}s)...")
    
    # Perform matrix operations to use memory and GPU
    x = jnp.ones((size, size), dtype=jnp.float32)
    y = jnp.ones((size, size), dtype=jnp.float32)
    
    # Take a snapshot during matrix creation
    take_memory_snapshot("During matrix creation")
    
    # Perform computation
    with monitor_memory_usage("Matrix Multiplication"):
        result = jnp.matmul(x, y)
        result.block_until_ready()
    
    # Simulate additional compute time
    time.sleep(compute_time)
    
    # Another operation with monitoring
    with monitor_memory_usage("Matrix Addition"):
        z = result + jnp.ones_like(result)
        z.block_until_ready()
    
    return z

def run_memory_test():
    """Run the memory monitoring test."""
    print("\n=== Memory Monitor Test ===\n")
    
    # Take initial snapshot
    initial_snapshot = take_memory_snapshot("Initial State")
    print_memory_snapshot(initial_snapshot)
    
    # Force allocation if needed
    print("\nChecking if memory allocation is needed...")
    allocation_result = force_allocation_if_needed(
        target_fraction=0.5,  # Try to reach 50% utilization
        current_usage_threshold=0.2,  # Only allocate if below 20%
        step_size_mb=1000,  # Allocate in 1GB steps
        max_steps=5  # Maximum 5 steps (5GB)
    )
    print(f"Allocation result: {json.dumps(allocation_result, indent=2)}")
    
    # Simulate training steps with increasing sizes
    for size in [2000, 4000, 6000]:
        with monitor_memory_usage(f"Training Step (size={size})"):
            simulate_training_step(size=size, compute_time=1.0)
    
    # Take final snapshot
    final_snapshot = take_memory_snapshot("Final State")
    print_memory_snapshot(final_snapshot)
    
    # Generate report
    print("\nGenerating memory report...")
    report = generate_memory_report("memory_test_report.json")
    
    # Print summary
    if "summary" in report and "gpu_utilization" in report["summary"]:
        util = report["summary"]["gpu_utilization"]
        print(f"\nTest Complete")
        print(f"GPU Utilization - Min: {util['min']:.2f}% Max: {util['max']:.2f}% Avg: {util['avg']:.2f}%")
    
    print("\nMemory test complete! Report saved to memory_test_report.json")

if __name__ == "__main__":
    run_memory_test() 