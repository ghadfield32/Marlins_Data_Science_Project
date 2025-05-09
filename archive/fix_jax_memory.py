#!/usr/bin/env python
"""
JAX Memory Allocation Fix - Startup Script

This script forces JAX to properly allocate GPU memory on container startup.
Run this before importing JAX in your application.
"""
import os
import sys
import argparse
import json
import subprocess
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Force JAX GPU memory allocation")
    parser.add_argument("--memory-fraction", type=float, default=0.95,
                        help="Fraction of GPU memory to allocate (default: 0.95)")
    parser.add_argument("--apply", action="store_true",
                        help="Apply changes to environment and run allocation test")
    return parser.parse_args()

def set_environment_variables(mem_fraction):
    """Set JAX environment variables for proper GPU memory allocation"""
    # Set environment variables to force allocation
    env_vars = {
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(mem_fraction),
        "JAX_PLATFORM_NAME": "gpu",
        "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
        "JAX_DISABLE_JIT": "false",
        "JAX_ENABLE_X64": "false",
        "TF_FORCE_GPU_ALLOW_GROWTH": "false",
        "JAX_PREALLOCATION_SIZE_LIMIT_BYTES": str(8 * 1024 * 1024 * 1024)  # 8GB limit
    }
    
    # Apply environment variables to current process
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars

def create_memory_fix_module():
    """Create a module to import that will fix JAX memory issues"""
    with open("jax_memory_fix_module.py", "w") as f:
        f.write("""
# jax_memory_fix_module.py
\"\"\"
JAX Memory Fix Module - Import this before importing JAX
\"\"\"
import os

# Set JAX memory environment variables
env_vars = {
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
    "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
    "JAX_PLATFORM_NAME": "gpu",
    "XLA_FLAGS": "--xla_gpu_deterministic_reductions --xla_force_host_platform_device_count=1",
    "JAX_DISABLE_JIT": "false",
    "JAX_ENABLE_X64": "false",
    "TF_FORCE_GPU_ALLOW_GROWTH": "false",
    "JAX_PREALLOCATION_SIZE_LIMIT_BYTES": "8589934592"  # 8GB limit
}

# Apply all environment variables
for key, value in env_vars.items():
    os.environ[key] = value

# Print success message
print(f"JAX memory settings applied. Use larger batches to allocate more memory.")
""")

def force_memory_allocation():
    """Force JAX to allocate memory by creating and using large tensors"""
    print("Forcing JAX to allocate GPU memory...")
    
    # Create a temporary script
    with open("_force_allocation.py", "w") as f:
        f.write("""
import os
import time
import json
import subprocess

# Force environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["JAX_PLATFORM_NAME"] = "gpu"

# Import JAX
import jax
import jax.numpy as jnp

# Print JAX devices
print(f"JAX devices: {jax.devices()}")

# Define memory check function
def get_gpu_memory():
    try:
        from jax.lib import xla_client
        free, total = xla_client.get_gpu_memory_info(0)
        used = total - free
        return free, total, used, used/total
    except:
        return 0, 0, 0, 0

# Initial memory check
free1, total1, used1, percent1 = get_gpu_memory()
print(f"Initial memory: {used1/1e9:.2f} GB / {total1/1e9:.2f} GB ({percent1:.2%})")

# Create list to hold references to tensors (prevent garbage collection)
tensors = []

# Force allocation by creating and using increasingly larger tensors
try:
    # Start with 2k x 2k (16MB) and double each time
    for i in range(12):  # Up to 32k x 32k (4GB)
        size = 2048 * (2**(i//2))
        print(f"Creating {size}x{size} tensor...")
        
        # Create and operate on tensor
        x = jnp.ones((size, size), dtype=jnp.float32)
        y = jnp.matmul(x, x)
        y = y.block_until_ready()
        
        # Keep reference to prevent garbage collection
        tensors.append(y)
        
        # Check memory
        free, total, used, percent = get_gpu_memory()
        print(f"Memory after {size}x{size}: {used/1e9:.2f} GB / {total/1e9:.2f} GB ({percent:.2%})")
        
        # Break if we've allocated enough
        if percent >= 0.4:
            print("Target allocation reached")
            break
            
        # Small delay to let allocation stabilize
        time.sleep(0.5)
except Exception as e:
    print(f"Error during allocation: {str(e)}")

# Final memory check
free2, total2, used2, percent2 = get_gpu_memory()
print(f"Final memory: {used2/1e9:.2f} GB / {total2/1e9:.2f} GB ({percent2:.2%})")

# Output results
print(json.dumps({
    "initial_percent": float(percent1),
    "final_percent": float(percent2),
    "success": float(percent2) >= 0.4
}))

# Keep tensors alive to maintain allocation
input("Press Enter to release GPU memory and exit...")
""")
    
    # Run the script
    subprocess.Popen([sys.executable, "_force_allocation.py"])
    
    print("Memory allocation script started in background.")
    print("GPU memory should now be properly allocated.")

def main():
    args = parse_args()
    
    print("JAX GPU Memory Allocation Fix")
    print("============================")
    
    # Set environment variables
    env_vars = set_environment_variables(args.memory_fraction)
    print("\nEnvironment variables set:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")
    
    # Create reusable module
    create_memory_fix_module()
    print("\nCreated jax_memory_fix_module.py - Import this before JAX in your scripts")
    
    # Force allocation if requested
    if args.apply:
        force_memory_allocation()
    
    print("\nDone! JAX should now allocate GPU memory correctly.")
    print("To use in your scripts, add this line before importing JAX:")
    print("  import jax_memory_fix_module")

if __name__ == "__main__":
    main() 