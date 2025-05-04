# Getting Started with JAX GPU in Docker

This guide will help you quickly set up and verify JAX with GPU support in Docker.

## Prerequisites

1. **Docker** installed and running
2. **NVIDIA GPU** with CUDA support
3. **NVIDIA Container Toolkit** (nvidia-docker2) installed
4. **Windows Subsystem for Linux 2 (WSL2)** if on Windows

## Quick Start

### 1. Clone this repository or download the files

### 2. Run the GPU verification scripts

Choose the appropriate script for your platform:

#### Windows (Command Prompt):
```cmd
cd gpu_testing
run_gpu_benchmark.cmd
```

#### Windows (PowerShell):
```powershell
cd gpu_testing
.\run_pymc_gpu.ps1
```

#### Linux/macOS:
```bash
cd gpu_testing
chmod +x run_jax_gpu.sh
./run_jax_gpu.sh
```

### 3. Check the results

If JAX is working with GPU support, you will see the GPU detected in the output and the matrix multiplication will be very fast (less than 0.5 seconds for a 5000x5000 matrix).

## Running Your Own JAX Scripts

To run your own JAX script with GPU support:

```bash
# Windows
docker run --rm --gpus all -it -v "%cd%":/app -w /app jax-gpu-test python3 your_script.py

# Linux/macOS
docker run --rm --gpus all -it -v "$(pwd)":/app -w /app jax-gpu-test python3 your_script.py
```

## Troubleshooting

1. **GPU not detected by JAX**: 
   - Make sure Docker has GPU access enabled
   - Verify GPU is visible with `nvidia-smi`
   - Check Docker Desktop settings to ensure GPU pass-through is enabled

2. **JAX shows `CpuDevice` but `jax.device_count('gpu')` is 1**:
   - This is expected behavior on some systems. JAX will still use the GPU for computations.
   - Check the speed of matrix multiplication to verify GPU acceleration is working.

3. **Docker container fails to start**:
   - Make sure nvidia-container-toolkit is correctly installed
   - Verify your NVIDIA drivers are up to date

## Example JAX GPU Code

Here's a simple snippet to verify GPU acceleration:

```python
import jax
import jax.numpy as jnp
from jax import random, jit
import time

# Print JAX information
print("JAX version:", jax.__version__)
print("JAX devices:", jax.devices())
print("GPU count:", jax.device_count("gpu"))

# Run a simple benchmark
key = random.key(0)
x = random.normal(key, (5000, 5000), dtype=jnp.float32)

@jit
def matmul(x):
    return jnp.dot(x, x.T)

# Warmup
_ = matmul(x).block_until_ready()

# Timed run
start = time.time()
result = matmul(x).block_until_ready()
elapsed = time.time() - start

print(f"Matrix multiplication time: {elapsed:.4f} seconds")
print(f"GPU acceleration working: {elapsed < 0.5}")
```

If the matrix multiplication takes less than 0.5 seconds, GPU acceleration is working! 