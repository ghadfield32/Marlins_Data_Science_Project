#!/usr/bin/env python
"""
Force JAX Memory Allocation

Utility to force JAX to allocate memory on the GPU, which helps with testing
memory management and ensuring memory limits are respected.

Usage:
    python bin/force_allocation.py [size_mb] [--keep]
    
    size_mb: Amount of memory to allocate in MB (default: 1000)
    --keep: Keep allocated memory (prevents garbage collection)

Example:
    python bin/force_allocation.py 2000 --keep  # Allocate 2GB and keep it
"""
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import memory management utilities
from src.utils.jax_memory_fix_module import (
    apply_jax_memory_fix, 
    force_memory_allocation,
    allocate_memory_incremental,
    verify_gpu_usage
)

def main():
    parser = argparse.ArgumentParser(description='Force JAX memory allocation')
    parser.add_argument('size_mb', nargs='?', type=int, default=1000,
                        help='Amount of memory to allocate in MB (default: 1000)')
    parser.add_argument('--keep', action='store_true',
                        help='Keep allocated memory (prevents garbage collection)')
    parser.add_argument('--incremental', action='store_true',
                        help='Allocate memory incrementally to target utilization')
    parser.add_argument('--target', type=float, default=0.6,
                        help='Target memory utilization fraction (0.0-1.0, default: 0.6)')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of allocation steps for incremental mode (default: 10)')
    parser.add_argument('--verify', action='store_true',
                        help='Only verify GPU usage, no allocation')
    
    args = parser.parse_args()
    
    # Apply memory settings first
    print("Applying JAX memory optimization settings...")
    apply_jax_memory_fix(fraction=0.95, preallocate=True, verbose=True)
    
    if args.verify:
        print("\nVerifying GPU usage...")
        result = verify_gpu_usage(minimal_usage=0.1, verbose=True)
        
        if result.get("success", False):
            print("GPU verification successful")
        else:
            print(f"GPU verification failed: {result.get('error', 'Unknown error')}")
        
        return
    
    if args.incremental:
        print(f"\nAllocating memory incrementally to reach {args.target:.0%} utilization...")
        result = allocate_memory_incremental(
            target_fraction=args.target,
            step_size_mb=args.size_mb,
            max_steps=args.steps,
            verbose=True
        )
        
        print(f"\nIncremental allocation complete:")
        print(f"Initial utilization: {result.get('initial_utilization', 0):.1%}")
        print(f"Final utilization: {result.get('final_utilization', 0):.1%}")
        print(f"Target reached: {result.get('target_reached', False)}")
        print(f"Steps taken: {result.get('steps_taken', 0)}")
        print(f"Memory allocated: {result.get('memory_allocated_mb', 0)} MB")
    else:
        print(f"\nAllocating {args.size_mb} MB of memory...")
        success = force_memory_allocation(
            size_mb=args.size_mb,
            keep_reference=args.keep,
            verbose=True
        )
        
        if success:
            print(f"Successfully allocated {args.size_mb} MB of memory")
            if args.keep:
                print("Memory will be kept allocated (use Python's garbage collection to release)")
            else:
                print("Memory will be released when this process ends")
        else:
            print("Failed to allocate memory")
    
if __name__ == '__main__':
    main() 