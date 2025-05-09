# JAX GPU Memory Optimization Module

This module provides tools to optimize JAX GPU memory usage in containerized environments. It addresses common issues with JAX memory allocation in Docker containers.

## Problem

JAX on GPU typically has issues with memory allocation in containerized environments:

1. By default, JAX may not properly preallocate GPU memory
2. Memory allocation can be unpredictable, leading to OOM errors
3. JAX 0.5.x with CUDA 12.x has different allocation behaviors than older versions

## Solution

The `jax_memory_fix_module.py` and `jax_gpu_utils.py` provide functions to:

1. Set optimal environment variables before JAX is imported
2. Force memory allocation when needed
3. Monitor GPU memory usage
4. Diagnose common memory configuration issues

## Usage

### Basic Usage

```python
# Import and apply memory settings before importing JAX
from src.utils.jax_memory_fix_module import apply_jax_memory_fix

# Configure with your settings
apply_jax_memory_fix(fraction=0.9, preallocate=True)

# Now it's safe to import JAX
import jax
import jax.numpy as jnp
```

### Forcing Memory Allocation

If you need to ensure memory is allocated upfront:

```python
from src.utils.jax_memory_fix_module import apply_jax_memory_fix, force_memory_allocation

# Apply memory settings
apply_jax_memory_fix()

# Force allocation of ~2GB
force_memory_allocation(size_mb=2000)

# Now import and use JAX
import jax
```

### Diagnosing Memory Issues

```python
from src.utils.jax_gpu_utils import log_gpu_diagnostics, check_jax_gpu_memory

# Log detailed GPU info
log_gpu_diagnostics()

# Check memory configuration and get recommendations
memory_check = check_jax_gpu_memory()
print(memory_check["recommendations"])
```

## Environment Variables

The module sets these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `true` | Whether to preallocate memory |
| `XLA_PYTHON_CLIENT_ALLOCATOR` | `platform` | Use platform's memory allocator |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.95` | Fraction of GPU memory to use |
| `JAX_PLATFORM_NAME` | `gpu` | Force JAX to use GPU |
| `XLA_FLAGS` | `--xla_force_host_platform_device_count=1` | XLA compiler flags |
| `JAX_DISABLE_JIT` | `false` | Keep JIT compilation enabled |
| `JAX_ENABLE_X64` | `false` | Use 32-bit precision by default |
| `JAX_PREALLOCATION_SIZE_LIMIT_BYTES` | `8589934592` | 8GB limit for preallocation |

## Expected Memory Usage

With JAX 0.5.2 and CUDA 12.x, you can expect:

1. Initial memory usage of ~6-7% with `XLA_PYTHON_CLIENT_PREALLOCATE=true`
2. Memory usage will grow as needed when operations are performed
3. You can force higher initial allocation using the `force_memory_allocation` function 