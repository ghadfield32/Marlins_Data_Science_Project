# src/utils/jax_memory_fix_module.py
"""
JAX Memory Fix Module - Import this before importing JAX

This module provides functions to optimize JAX GPU memory usage by setting
appropriate environment variables before JAX is imported.

Usage:
    # Option 1: Import the module directly (applies default settings)
    import jax_memory_fix_module
    import jax
    
    # Option 2: Import and explicitly apply settings
    from jax_memory_fix_module import apply_jax_memory_fix
    apply_jax_memory_fix(fraction=0.9)
    import jax
"""
import os
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

# Define JAX memory environment variables
env_vars = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    "JAX_PLATFORM_NAME": "gpu",
    "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
    "JAX_DISABLE_JIT": "false",
    "JAX_ENABLE_X64": "false",
    "TF_FORCE_GPU_ALLOW_GROWTH": "false",
    "JAX_PREALLOCATION_SIZE_LIMIT_BYTES": "8589934592"  # 8GB limit
}

# Flag to track if settings have been applied
_settings_applied = False

# Global registry to keep references to allocated tensors
# This prevents garbage collection and ensures memory stays allocated
_memory_registry = {}

def apply_jax_memory_fix(fraction=0.95, preallocate=True, verbose=True, 
                          prealloc_limit_bytes=None):
    """
    Apply JAX GPU memory optimization settings.
    
    This function sets environment variables to optimize JAX's GPU memory usage.
    
    Args:
        fraction (float): Fraction of GPU memory that JAX can use (0.0 to 1.0)
        preallocate (bool): Whether to preallocate GPU memory
        verbose (bool): Whether to print diagnostic information
        prealloc_limit_bytes (int, optional): Limit for preallocation in bytes
        
    Returns:
        dict: The applied environment variables
    """
    global _settings_applied
    
    # Update with user-provided values
    settings = env_vars.copy()
    settings["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(fraction)
    settings["XLA_PYTHON_CLIENT_PREALLOCATE"] = str(preallocate).lower()
    
    # Set preallocation limit if provided
    if prealloc_limit_bytes is not None:
        settings["JAX_PREALLOCATION_SIZE_LIMIT_BYTES"] = str(prealloc_limit_bytes)
    
    # Apply all environment variables
    for key, value in settings.items():
        os.environ[key] = value
    
    if verbose and not _settings_applied:
        print(f"JAX memory settings applied (fraction={fraction}, preallocate={preallocate})")
        print(f"Use larger batches to force more memory allocation if needed.")
    
    _settings_applied = True
    return settings

def force_memory_allocation(size_mb=1000, verbose=True, keep_reference=True, 
                           dtype='float32', operation='matmul'):
    """
    Force JAX to allocate GPU memory by creating and using a large tensor.
    
    Note: This function imports JAX, so it should only be called after
    apply_jax_memory_fix() if you want to ensure settings are applied first.
    
    Args:
        size_mb (int): Amount of memory to allocate in MB
        verbose (bool): Whether to print progress messages
        keep_reference (bool): Whether to keep a reference to prevent garbage collection
        dtype (str): Data type for the tensor ('float32', 'float16', or 'bfloat16')
        operation (str): Operation to perform on tensor ('matmul', 'add', 'none')
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import jax
        import jax.numpy as jnp
        
        # Map dtype string to JAX dtype
        dtype_map = {
            'float32': jnp.float32,
            'float16': jnp.float16,
            'bfloat16': jnp.bfloat16
        }
        
        if dtype not in dtype_map:
            if verbose:
                print(f"Warning: Unsupported dtype '{dtype}'. Using float32 instead.")
            jax_dtype = jnp.float32
        else:
            jax_dtype = dtype_map[dtype]
        
        # Calculate bytes per element
        bytes_per_element = 4 if dtype == 'float32' else 2
        
        # Create a matrix that will use approximately size_mb of memory
        n = int(pow(size_mb * 1024 * 1024 / bytes_per_element, 0.5))
        
        if verbose:
            print(f"Allocating ~{size_mb} MB with {n}x{n} {dtype} matrix...")
        
        # Create tensor
        x = jnp.ones((n, n), dtype=jax_dtype)
        
        # Perform operation based on specified type
        if operation == 'matmul':
            y = jnp.matmul(x, x)
        elif operation == 'add':
            y = x + x
        else:  # 'none'
            y = x
            
        # Ensure operation completes
        y.block_until_ready()
        
        # Store reference to prevent garbage collection if requested
        if keep_reference:
            global _memory_registry
            allocation_id = f"tensor_{len(_memory_registry)}"
            _memory_registry[allocation_id] = y
            
            if verbose:
                print(f"Successfully allocated ~{size_mb} MB of GPU memory (ID: {allocation_id})")
        else:
            if verbose:
                print(f"Successfully allocated ~{size_mb} MB of GPU memory (temporary)")
        
        return True
    except Exception as e:
        if verbose:
            print(f"Error allocating memory: {str(e)}")
        return False

def allocate_memory_incremental(target_fraction=0.6, step_size_mb=1000, 
                               max_steps=10, check_interval=0.5, verbose=True):
    """
    Incrementally allocate GPU memory until reaching a target utilization.
    
    Args:
        target_fraction (float): Target fraction of GPU memory to use (0.0 to 1.0)
        step_size_mb (int): Size of each allocation step in MB
        max_steps (int): Maximum number of allocation steps
        check_interval (float): Time in seconds to wait between allocations
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Information about the allocation process
    """
    try:
        # Import here to avoid circular imports
        from src.utils.jax_gpu_utils import get_gpu_memory_info
        
        # Get initial memory info
        memory_info = get_gpu_memory_info()
        if not memory_info or "nvidia_smi" not in memory_info:
            if verbose:
                print("Error: Could not get GPU memory info")
            return {"success": False, "error": "Could not get GPU memory info"}
        
        # Get initial utilization
        gpu_info = memory_info["nvidia_smi"][0]  # Use first GPU
        initial_utilization = gpu_info["utilization"] / 100.0
        current_utilization = initial_utilization
        
        if verbose:
            print(f"Initial GPU memory utilization: {initial_utilization:.2%}")
            print(f"Target utilization: {target_fraction:.2%}")
        
        # Allocate in steps
        allocations = []
        for step in range(max_steps):
            if current_utilization >= target_fraction:
                if verbose:
                    print(f"Target utilization reached: {current_utilization:.2%}")
                break
                
            # Allocate one step
            allocation_id = f"step_{step}"
            success = force_memory_allocation(
                size_mb=step_size_mb,
                verbose=verbose,
                keep_reference=True
            )
            
            if not success:
                if verbose:
                    print(f"Allocation failed at step {step}")
                break
                
            allocations.append(allocation_id)
            
            # Wait for memory status to stabilize
            time.sleep(check_interval)
            
            # Check new utilization
            memory_info = get_gpu_memory_info()
            if memory_info and "nvidia_smi" in memory_info:
                gpu_info = memory_info["nvidia_smi"][0]
                current_utilization = gpu_info["utilization"] / 100.0
                
                if verbose:
                    print(f"Step {step+1}: Utilization now {current_utilization:.2%}")
        
        return {
            "success": True,
            "initial_utilization": initial_utilization,
            "final_utilization": current_utilization,
            "target_reached": current_utilization >= target_fraction,
            "steps_taken": len(allocations),
            "memory_allocated_mb": len(allocations) * step_size_mb
        }
    except Exception as e:
        if verbose:
            print(f"Error in incremental allocation: {str(e)}")
        return {"success": False, "error": str(e)}

def verify_gpu_usage(minimal_usage=0.1, operation_size=5000, verbose=True):
    """
    Verify that JAX is using the GPU and allocating memory properly.
    
    Args:
        minimal_usage (float): Minimum acceptable memory usage fraction
        operation_size (int): Size of test operations
        verbose (bool): Whether to print progress
        
    Returns:
        dict: Verification results including success status and memory info
    """
    try:
        import jax
        import jax.numpy as jnp
        from src.utils.jax_gpu_utils import get_gpu_memory_info
        
        # Check for GPU devices
        devices = jax.devices()
        has_gpu = any(d.platform == "gpu" for d in devices)
        
        if not has_gpu:
            if verbose:
                print("No GPU devices available to JAX")
            return {
                "success": False,
                "has_gpu": False,
                "error": "No GPU devices available to JAX"
            }
        
        # Get initial memory info
        initial_mem = get_gpu_memory_info()
        if not initial_mem or "nvidia_smi" not in initial_mem:
            if verbose:
                print("Error: Could not get GPU memory info")
            return {"success": False, "error": "Could not get GPU memory info"}
        
        initial_usage = initial_mem["nvidia_smi"][0]["utilization"] / 100.0
        
        if verbose:
            print(f"Initial GPU memory usage: {initial_usage:.2%}")
        
        # Run a test operation
        if verbose:
            print(f"Running test operation (size: {operation_size}x{operation_size})...")
            
        # Time the operation to verify GPU is working
        start_time = time.time()
        x = jnp.ones((operation_size, operation_size), dtype=jnp.float32)
        y = jnp.matmul(x, x)
        y.block_until_ready()
        operation_time = time.time() - start_time
        
        # Get memory after operation
        time.sleep(0.5)  # Wait for memory status to stabilize
        final_mem = get_gpu_memory_info()
        final_usage = final_mem["nvidia_smi"][0]["utilization"] / 100.0
        
        # Calculate memory increase
        mem_increase = final_usage - initial_usage
        
        # Verify minimum usage
        usage_ok = final_usage >= minimal_usage
        
        if verbose:
            print(f"Final GPU memory usage: {final_usage:.2%}")
            print(f"Memory increase: {mem_increase:.2%}")
            print(f"Operation time: {operation_time:.4f} seconds")
            print(f"Minimum usage requirement met: {usage_ok}")
        
        # Advanced GPU check using JAX's own reporting if available
        try:
            from jax.lib import xla_client as xc
            if hasattr(xc, "get_gpu_memory_info"):
                free, total = xc.get_gpu_memory_info(0)
                used = total - free
                jax_usage = used / total
                
                if verbose:
                    print(f"JAX-reported GPU memory usage: {jax_usage:.2%}")
                    
                return {
                    "success": usage_ok,
                    "has_gpu": True,
                    "initial_usage": initial_usage,
                    "final_usage": final_usage,
                    "jax_reported_usage": jax_usage,
                    "memory_increase": mem_increase,
                    "operation_time": operation_time,
                    "usage_ok": usage_ok
                }
        except Exception as e:
            if verbose:
                print(f"Could not get JAX-specific memory info: {e}")
        
        return {
            "success": usage_ok,
            "has_gpu": True,
            "initial_usage": initial_usage,
            "final_usage": final_usage,
            "memory_increase": mem_increase,
            "operation_time": operation_time,
            "usage_ok": usage_ok
        }
    except Exception as e:
        if verbose:
            print(f"Error verifying GPU usage: {str(e)}")
        return {"success": False, "error": str(e)}

def clear_memory_registry(verbose=True):
    """
    Clear all allocated tensors from the memory registry to free up GPU memory.
    
    Args:
        verbose (bool): Whether to print information
        
    Returns:
        int: Number of tensors cleared
    """
    global _memory_registry
    count = len(_memory_registry)
    
    if verbose and count > 0:
        print(f"Clearing {count} tensors from memory registry...")
        
    _memory_registry.clear()
    
    # Try to force garbage collection
    try:
        import gc
        gc.collect()
    except:
        pass
        
    return count

def get_memory_registry_info():
    """
    Get information about tensors in the memory registry.
    
    Returns:
        dict: Information about registered tensors
    """
    global _memory_registry
    return {
        "count": len(_memory_registry),
        "keys": list(_memory_registry.keys())
    }

# Only apply settings on import if they haven't been applied yet
if not _settings_applied:
    apply_jax_memory_fix(verbose=True)
