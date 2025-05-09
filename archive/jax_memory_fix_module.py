# jax_memory_fix_module.py
"""
JAX Memory Fix Module - Import this before importing JAX
"""
import os

# Set JAX memory environment variables
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

# Apply all environment variables
for key, value in env_vars.items():
    os.environ[key] = value

# Print success message
print(f"JAX memory settings applied. Use larger batches to allocate more memory.")
