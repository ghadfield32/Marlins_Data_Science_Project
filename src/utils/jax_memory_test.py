#!/usr/bin/env python
"""
JAX Memory Testing and Verification Module

This module provides tools to test and verify JAX GPU memory usage.
It includes benchmarks and verification utilities to ensure JAX is
properly utilizing GPU memory.
"""
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def benchmark_memory_operations(sizes=(1000, 2000, 4000, 8000), 
                               dtypes=('float32', 'float16', 'bfloat16'),
                               verbose=True):
    """
    Benchmark JAX memory operations with different sizes and datatypes.
    
    Args:
        sizes (tuple): Matrix sizes to test
        dtypes (tuple): Data types to test
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Benchmark results
    """
    try:
        # Import memory modules
        from src.utils.jax_memory_fix_module import force_memory_allocation, clear_memory_registry
        from src.utils.jax_gpu_utils import get_gpu_memory_info
        import jax
        import jax.numpy as jnp
        
        # Print JAX info
        if verbose:
            print(f"JAX version: {jax.__version__}")
            print(f"JAX devices: {jax.devices()}")
            print(f"Starting memory benchmark...")
            
        results = {}
        
        # Run benchmarks for each combination
        for dtype in dtypes:
            dtype_results = {}
            for size in sizes:
                if verbose:
                    print(f"\nTesting size={size}x{size}, dtype={dtype}")
                    
                # Clear previous allocations
                clear_memory_registry()
                
                # Get initial memory
                initial_mem = get_gpu_memory_info()
                if initial_mem and "nvidia_smi" in initial_mem:
                    initial_usage = initial_mem["nvidia_smi"][0]["utilization"] / 100.0
                else:
                    initial_usage = 0
                    
                # Measure time to allocate and run operation
                start_time = time.time()
                force_memory_allocation(
                    size_mb=size, 
                    dtype=dtype,
                    operation='matmul',
                    verbose=False
                )
                allocation_time = time.time() - start_time
                
                # Get final memory
                time.sleep(0.5)  # Let memory usage stabilize
                final_mem = get_gpu_memory_info()
                if final_mem and "nvidia_smi" in final_mem:
                    final_usage = final_mem["nvidia_smi"][0]["utilization"] / 100.0
                else:
                    final_usage = 0
                    
                memory_increase = final_usage - initial_usage
                
                # Store results
                test_result = {
                    "initial_usage": initial_usage,
                    "final_usage": final_usage,
                    "memory_increase": memory_increase,
                    "allocation_time": allocation_time
                }
                
                if verbose:
                    print(f"  Initial memory usage: {initial_usage:.2%}")
                    print(f"  Final memory usage: {final_usage:.2%}")
                    print(f"  Memory increase: {memory_increase:.2%}")
                    print(f"  Allocation time: {allocation_time:.4f} seconds")
                
                dtype_results[size] = test_result
            
            results[dtype] = dtype_results
        
        # Clear all allocations
        clear_memory_registry()
        
        return {
            "success": True,
            "results": results,
            "summary": generate_benchmark_summary(results)
        }
    except Exception as e:
        logger.error(f"Error in benchmark: {e}")
        return {"success": False, "error": str(e)}

def generate_benchmark_summary(results):
    """
    Generate a summary of benchmark results.
    
    Args:
        results (dict): Benchmark results
        
    Returns:
        dict: Summary of benchmark results
    """
    summary = {"by_dtype": {}, "by_size": {}}
    
    # Skip if results is empty
    if not results:
        return summary
    
    # Calculate averages by dtype
    for dtype, size_results in results.items():
        avg_increase = sum(r["memory_increase"] for r in size_results.values()) / len(size_results)
        avg_time = sum(r["allocation_time"] for r in size_results.values()) / len(size_results)
        summary["by_dtype"][dtype] = {
            "avg_memory_increase": avg_increase,
            "avg_allocation_time": avg_time
        }
    
    # Calculate averages by size
    sizes = list(next(iter(results.values())).keys())
    for size in sizes:
        avg_increase = sum(results[dt][size]["memory_increase"] for dt in results.keys()) / len(results)
        avg_time = sum(results[dt][size]["allocation_time"] for dt in results.keys()) / len(results)
        summary["by_size"][size] = {
            "avg_memory_increase": avg_increase,
            "avg_allocation_time": avg_time
        }
    
    return summary

def test_gpu_scaling(max_percentage=0.8, steps=5, verbose=True):
    """
    Test how JAX scales memory usage with increasing workload sizes.
    
    Args:
        max_percentage (float): Maximum percentage of GPU memory to target
        steps (int): Number of steps to test
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Test results
    """
    try:
        from src.utils.jax_memory_fix_module import (
            force_memory_allocation, 
            clear_memory_registry,
            allocate_memory_incremental
        )
        
        # Clear any existing allocations
        clear_memory_registry()
        
        if verbose:
            print(f"Testing GPU memory scaling in {steps} steps up to {max_percentage:.0%} usage")
        
        # Perform incremental allocation
        result = allocate_memory_incremental(
            target_fraction=max_percentage,
            step_size_mb=1000,
            max_steps=steps,
            check_interval=0.5,
            verbose=verbose
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in scaling test: {e}")
        return {"success": False, "error": str(e)}

def verify_jax_gpu_performance(matrix_size=5000, num_operations=3, verbose=True):
    """
    Verify JAX is using the GPU by measuring operation speed.
    
    Args:
        matrix_size (int): Size of test matrices
        num_operations (int): Number of operations to perform
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Performance test results
    """
    try:
        import jax
        import jax.numpy as jnp
        
        # Get device info
        devices = jax.devices()
        device_info = [str(d) for d in devices]
        
        if verbose:
            print(f"Testing JAX performance on devices: {device_info}")
            print(f"Matrix size: {matrix_size}x{matrix_size}")
            
        # Create test matrices
        a = jnp.ones((matrix_size, matrix_size), dtype=jnp.float32)
        b = jnp.ones((matrix_size, matrix_size), dtype=jnp.float32)
        
        # Warmup
        if verbose:
            print("Performing warmup operation...")
        _ = jnp.matmul(a, b).block_until_ready()
        
        # Measure operation time
        times = []
        for i in range(num_operations):
            if verbose:
                print(f"Operation {i+1}/{num_operations}...")
                
            start_time = time.time()
            c = jnp.matmul(a, b)
            c.block_until_ready()
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        if verbose:
            print(f"Average operation time: {avg_time:.4f} seconds")
            
        # Check if we're likely using GPU
        # A large matrix multiplication should be very fast on GPU
        likely_gpu = avg_time < 1.0  # Rough heuristic
        
        return {
            "success": True,
            "devices": device_info,
            "operation_times": times,
            "average_time": avg_time,
            "likely_using_gpu": likely_gpu
        }
    except Exception as e:
        logger.error(f"Error in performance test: {e}")
        return {"success": False, "error": str(e)}

def run_comprehensive_test(verbose=True):
    """
    Run a comprehensive test of JAX GPU memory and performance.
    
    Args:
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Comprehensive test results
    """
    from src.utils.jax_memory_fix_module import (
        apply_jax_memory_fix, 
        force_memory_allocation,
        clear_memory_registry,
        verify_gpu_usage
    )
    
    results = {}
    
    if verbose:
        print("\n=== JAX GPU Memory Comprehensive Test ===\n")
    
    # 1. Apply memory fix
    if verbose:
        print("1. Applying JAX memory optimization settings...")
    apply_jax_memory_fix(fraction=0.9, preallocate=True, verbose=verbose)
    
    # 2. Verify GPU availability and usage
    if verbose:
        print("\n2. Verifying GPU availability and memory usage...")
    results["gpu_verification"] = verify_gpu_usage(
        minimal_usage=0.05,
        operation_size=5000,
        verbose=verbose
    )
    
    # 3. Test a simple memory allocation
    if verbose:
        print("\n3. Testing simple memory allocation...")
    force_memory_allocation(size_mb=2000, verbose=verbose)
    
    # 4. Performance test
    if verbose:
        print("\n4. Testing GPU performance...")
    results["performance_test"] = verify_jax_gpu_performance(
        matrix_size=5000,
        verbose=verbose
    )
    
    # 5. Memory scaling test
    if verbose:
        print("\n5. Testing memory scaling...")
    results["scaling_test"] = test_gpu_scaling(
        max_percentage=0.5,  # Use 50% as max to avoid OOM
        steps=3,
        verbose=verbose
    )
    
    # 6. Run benchmarks (limited)
    if verbose:
        print("\n6. Running memory benchmarks...")
    results["benchmarks"] = benchmark_memory_operations(
        sizes=(2000, 4000),
        dtypes=('float32', 'float16'),
        verbose=verbose
    )
    
    # 7. Clean up
    if verbose:
        print("\n7. Cleaning up...")
    clear_memory_registry()
    
    if verbose:
        print("\n=== Test Complete ===")
    
    # Overall success determination
    results["overall_success"] = (
        results.get("gpu_verification", {}).get("success", False) and
        results.get("performance_test", {}).get("success", False) and
        results.get("scaling_test", {}).get("success", False) and
        results.get("benchmarks", {}).get("success", False)
    )
    
    return results

if __name__ == "__main__":
    # Run comprehensive test when executed directly
    results = run_comprehensive_test(verbose=True)
    
    # Save results to file
    with open("jax_memory_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nResults saved to jax_memory_test_results.json")
    
    # Print overall success
    if results["overall_success"]:
        print("\n✅ All tests passed: JAX is properly utilizing GPU memory!")
    else:
        print("\n❌ Some tests failed: See detailed results for more information.") 