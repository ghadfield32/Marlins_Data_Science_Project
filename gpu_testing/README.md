# JAX GPU Setup with Docker on Windows

This document summarizes the findings from our investigation into setting up JAX with GPU support in a Docker container on Windows.

## Quick Start

We've created several scripts to make it easy to test and use JAX with GPU support:

### Windows Users
- `run_jax_gpu.bat` - Basic JAX GPU test and information
- `run_gpu_benchmark.cmd` - Run a matrix multiplication benchmark to verify GPU speed
- `run_pymc_gpu.ps1` (PowerShell) - Run PyMC with JAX backend on GPU

### Linux/macOS Users
- `run_jax_gpu.sh` - Basic JAX GPU test and information

Simply run the appropriate script for your platform to verify GPU support works correctly.

## Using the JAX GPU Container

To run your own JAX scripts with GPU support:

```bash
# Windows
docker run --rm --gpus all -it -v "%cd%":/app -w /app jax-gpu-test python3 your_script.py

# Linux/macOS
docker run --rm --gpus all -it -v "$(pwd)":/app -w /app jax-gpu-test python3 your_script.py
```

## Key Findings

1. **GPU Detection Issues**:
   - JAX has inconsistent GPU detection behavior in Docker containers on Windows
   - `jax.devices()` might show only CPU devices even when GPU is available
   - `jax.device_count('gpu')` can correctly report GPU availability even when `jax.devices()` doesn't show it

2. **Working Configuration**:
   - Base image: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
   - JAX version: ≥ 0.4.23
   - Install JAX directly (not in a virtualenv)
   - The test container `jax-gpu-test` correctly detects and uses the GPU

3. **Docker Commands**:
   - Running with GPU access: `docker run --rm --gpus all -it jax-gpu-test`
   - When running your scripts: `docker run --rm --gpus all -it -v ${PWD}:/app -w /app jax-gpu-test python3 your_script.py`

## Testing GPU Availability

To verify JAX can see the GPU, run:

```python
import jax
print('JAX Devices:', jax.devices())
print('GPU count:', jax.device_count('gpu'))
```

If `jax.device_count('gpu')` reports a count > 0, GPU acceleration should work even if devices only shows CPU.

## Matrix Multiplication Benchmark

To test actual GPU performance:

```python
import jax
import jax.numpy as jnp
from jax import random, jit
import time

key = random.key(0)
x = random.normal(key, (5000, 5000), dtype=jnp.float32)

@jit
def matmul(x):
    return jnp.dot(x, x.T)

# Warmup run (to compile)
result = matmul(x)
result.block_until_ready()

# Timed run
start_time = time.time()
result = matmul(x)
result.block_until_ready()  # Wait for computation to complete
elapsed = time.time() - start_time

print(f"Matrix multiplication time: {elapsed:.4f} seconds")
```

On a GPU, this should complete in milliseconds rather than seconds.

## Running PyMC with JAX Backend

Set these environment variables:
- `PYTENSOR_FLAGS=mode=JAX,floatX=float32`
- `JAX_PLATFORMS=cpu,cuda`
- `XLA_FLAGS=--xla_gpu_force_compilation_parallelism=1`

Then import PyMC:
```python
import pymc as pm
import pytensor
print('PyTensor mode:', pytensor.config.mode)
```

When running on Windows with Docker, prefer direct `docker run` commands with GPU passthrough over docker-compose. 