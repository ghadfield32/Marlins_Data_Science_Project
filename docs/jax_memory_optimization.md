# JAX GPU Memory Optimization

This document describes the JAX GPU memory optimization setup used in this project, which helps avoid memory fragmentation and OOM errors when using JAX with GPUs.

## Background

JAX can experience memory issues on GPU for several reasons:

1. **Memory Fragmentation**: JAX's default behavior allows memory to become fragmented
2. **Lack of Preallocation**: Without preallocation, JAX allocates memory as needed, which can lead to OOM errors
3. **No Memory Limit**: Without a limit, JAX may try to use all available GPU memory

## Environment Variables

Our implementation sets these important environment variables:

| Variable | Value | Purpose |
|----------|-------|---------|
| `XLA_PYTHON_CLIENT_PREALLOCATE` | `true` | Pre-allocates a memory pool on startup |
| `XLA_PYTHON_CLIENT_ALLOCATOR` | `platform` | Uses the platform allocator (usually more efficient) |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.95` | Limits JAX to use 95% of total GPU memory |
| `JAX_PLATFORM_NAME` | `gpu` | Forces JAX to use GPU instead of CPU |
| `XLA_FLAGS` | `--xla_force_host_platform_device_count=1` | Controls device visibility |
| `JAX_DISABLE_JIT` | `false` | Enables JIT compilation for better performance |
| `JAX_ENABLE_X64` | `false` | Disables 64-bit mode to save memory |
| `JAX_PREALLOCATION_SIZE_LIMIT_BYTES` | `8589934592` | Limits preallocation to 8GB |

## Integration in Docker Environment

These settings are applied in multiple places to ensure consistency:

1. **Dockerfile**: Environment variables are set during container build
2. **docker-compose.yml**: Environment variables are set for the container
3. **devcontainer.env**: Environment variables for VS Code devcontainer
4. **Memory Fix Module**: `jax_memory_fix_module.py` provides programmatic control

## Memory Fix Module Usage

The `jax_memory_fix_module.py` should be imported **before** importing JAX:

```python
# Option 1: Import the module directly (applies default settings)
import src.utils.jax_memory_fix_module
import jax

# Option 2: Import and explicitly apply settings
from src.utils.jax_memory_fix_module import apply_jax_memory_fix
apply_jax_memory_fix(fraction=0.9)
import jax
```

## Utility Functions

The module provides several utility functions:

1. `apply_jax_memory_fix()`: Sets environment variables before JAX is imported
2. `force_memory_allocation()`: Forces JAX to allocate memory for testing
3. `allocate_memory_incremental()`: Incrementally allocates memory to target utilization
4. `verify_gpu_usage()`: Verifies JAX is using the GPU correctly
5. `clear_memory_registry()`: Clears allocated tensors to free memory

## Utility Scripts

The project includes utility scripts to help with memory management:

1. **bin/monitor_gpu.py**: Monitors GPU memory usage in real-time
2. **bin/force_allocation.py**: Forces JAX to allocate memory for testing
3. **.devcontainer/gpu_verify.py**: Verifies GPU is working at container startup

## Memory Management Best Practices

When working with JAX and GPU memory:

1. **Import Order**: Always import memory fix module before importing JAX
2. **Batch Size**: Use appropriate batch sizes; too large can cause OOM errors
3. **Tensor Types**: Use lower precision (e.g., bfloat16 or float16) when possible
4. **Clear Cache**: Use `clear_memory_registry()` to manually free memory
5. **Monitor Usage**: Use `bin/monitor_gpu.py` to monitor memory usage
6. **Limit Preallocation**: Use the preallocation limit for large models

## Debugging Memory Issues

If you encounter memory issues:

1. Run `bin/monitor_gpu.py` to see real-time memory usage
2. Check if preallocation is enabled with `echo $XLA_PYTHON_CLIENT_PREALLOCATE`
3. Verify JAX is using GPU with `bin/force_allocation.py --verify`
4. Try reducing batch size or model complexity
5. Use explicit garbage collection: `import gc; gc.collect()`

## Example Notebook

See `examples/jax_memory_demo.ipynb` for a complete demonstration of memory management techniques. 