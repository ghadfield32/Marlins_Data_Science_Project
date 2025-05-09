#!/usr/bin/env python
"""
Test JAX GPU memory allocation
"""
import os
import json
import re
import subprocess
import time

# Set environment variables BEFORE importing JAX
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["JAX_PLATFORM_NAME"] = "gpu"

# Print the environment variables to verify they're set
print("Environment variables:")
for k in ["XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_ALLOCATOR", 
          "XLA_PYTHON_CLIENT_MEM_FRACTION", "JAX_PLATFORM_NAME"]:
    print(f"  {k}={os.environ.get(k, 'Not set')}")

# Now import JAX
print("\nImporting JAX...")
import jax
import jax.numpy as jnp
from jax.lib import xla_client as xc

# Check JAX version and devices
print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")

# Define function to check GPU memory
def check_gpu_memory():
    if hasattr(xc, "get_gpu_memory_info"):
        free, total = xc.get_gpu_memory_info(0)
    else:
        # Fall back to nvidia-smi
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"], text=True).splitlines()[0]
        total, free = (int(s.strip()) for s in re.split(r",\s*", output, maxsplit=1))
        free *= 1_048_576  # MiB to bytes
        total *= 1_048_576  # MiB to bytes
    
    used = total - free
    used_percent = used / total
    return free, total, used, used_percent

# Check memory before operations
free1, total1, used1, percent1 = check_gpu_memory()
print(f"\nBefore operations:")
print(f"  Total GPU memory: {total1/1e9:.2f} GB")
print(f"  Free GPU memory: {free1/1e9:.2f} GB")
print(f"  Used GPU memory: {used1/1e9:.2f} GB ({percent1:.2%})")

# Run an operation to allocate memory
print("\nRunning a matrix operation...")
x = jnp.ones((5000, 5000), dtype=jnp.float32)  # 100MB matrix
y = jnp.dot(x, x)  # Matrix multiplication
y.block_until_ready()  # Ensure operation completes

# Check memory after operations
time.sleep(1)  # Give time for memory to stabilize
free2, total2, used2, percent2 = check_gpu_memory()
print(f"\nAfter operations:")
print(f"  Total GPU memory: {total2/1e9:.2f} GB")
print(f"  Free GPU memory: {free2/1e9:.2f} GB")
print(f"  Used GPU memory: {used2/1e9:.2f} GB ({percent2:.2%})")
print(f"  Memory difference: {(used2-used1)/1e9:.2f} GB ({(percent2-percent1):.2%})")

# Try larger matrix to force more allocation
print("\nRunning a larger matrix operation...")
big_x = jnp.ones((10000, 10000), dtype=jnp.float32)  # 400MB matrix
big_y = jnp.dot(big_x, big_x)  # Matrix multiplication
big_y.block_until_ready()  # Ensure operation completes

# Check memory after large operations
time.sleep(1)  # Give time for memory to stabilize
free3, total3, used3, percent3 = check_gpu_memory()
print(f"\nAfter large operations:")
print(f"  Total GPU memory: {total3/1e9:.2f} GB")
print(f"  Free GPU memory: {free3/1e9:.2f} GB")
print(f"  Used GPU memory: {used3/1e9:.2f} GB ({percent3:.2%})")
print(f"  Memory difference from start: {(used3-used1)/1e9:.2f} GB ({(percent3-percent1):.2%})")

# Output JSON for potential test integration
result = {
    "jax_version": jax.__version__,
    "pre": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
    "frac": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
    "allocator": os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR"),
    "initial_pool": percent1,
    "after_ops_pool": percent2,
    "after_large_ops_pool": percent3,
    "need": 0.4 if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "true" else 0.05
}
print(f"\nJSON result: {json.dumps(result)}") 