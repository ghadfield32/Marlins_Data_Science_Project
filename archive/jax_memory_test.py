import os
import time
import json
import subprocess
import re

# Force environment variables
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["JAX_PLATFORM_NAME"] = "gpu"

# Now import JAX
import jax
import jax.numpy as jnp
from jax.lib import xla_client as xc

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
print(f"Initial memory usage: {used1/1e9:.2f} GB ({percent1:.2%})")

# Force memory allocation by creating large tensors
print("Creating tensors to force memory allocation...")
tensors = []
for i in range(10):
    # Create a 1GB tensor each iteration
    x = jnp.ones((8192, 8192), dtype=jnp.float32)
    y = jnp.matmul(x, x)
    y.block_until_ready()
    tensors.append(y)  # Keep reference to prevent garbage collection
    
    # Check memory
    free_now, total_now, used_now, percent_now = check_gpu_memory()
    print(f"Iteration {i+1}: Using {used_now/1e9:.2f} GB ({percent_now:.2%})")
    
    # Break if we've allocated enough
    if percent_now >= 0.4:
        print(f"Successfully allocated {percent_now:.2%} of GPU memory")
        break

# Final memory check
free2, total2, used2, percent2 = check_gpu_memory()

# Output results
result = {
    "initial_percent": percent1,
    "final_percent": percent2,
    "success": percent2 >= 0.4
}
print(json.dumps(result)) 