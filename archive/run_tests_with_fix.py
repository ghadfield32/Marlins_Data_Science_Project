#!/usr/bin/env python
"""
Run tests with memory fix applied
"""
import os
import sys
import subprocess
import time

# Set environment variables for JAX memory allocation
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["JAX_DISABLE_JIT"] = "false"
os.environ["JAX_ENABLE_X64"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false"
os.environ["JAX_PREALLOCATION_SIZE_LIMIT_BYTES"] = str(8 * 1024 * 1024 * 1024)  # 8GB limit

print("JAX memory settings applied:")
for key, value in os.environ.items():
    if key.startswith(("XLA_", "JAX_", "TF_")):
        print(f"  {key}={value}")

# Force memory allocation
print("\nForcing memory allocation...")
# Create a temporary script
with open("_allocate_memory.py", "w") as f:
    f.write("""
import os
import jax
import jax.numpy as jnp
import time

# Allocate memory with large tensors
tensors = []
for i in range(5):
    x = jnp.ones((8192, 8192), dtype=jnp.float32)
    y = jnp.matmul(x, x)
    y.block_until_ready()
    tensors.append(y)  # Keep reference to prevent GC
    
print(f"Allocated memory with {len(tensors)} large tensors")
time.sleep(2)  # Let memory stabilize
""")

# Run memory allocation
subprocess.run([sys.executable, "_allocate_memory.py"])

# Run the tests
print("\nRunning tests...")
test_cmd = ["pytest", "-v", "tests/test_jax_memory.py"]
subprocess.run(test_cmd)

# Clean up
try:
    os.remove("_allocate_memory.py")
except:
    pass

print("\nDone!") 