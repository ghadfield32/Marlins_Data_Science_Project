---
name: JAX cuDNN Version Mismatch
about: Report a cuDNN/JAX version incompatibility causing silent crashes
title: "JAX 0.5.2 crashes with cuDNN 9.0.0 (compiled against 9.1.1)"
labels: bug, cudnn
assignees: ''
---

## Environment Information

### JAX/CUDA Configuration
- JAX version: 0.5.2
- JAXlib version: 0.5.1
- Python version: 3.10.17
- GPU: NVIDIA GeForce RTX 5080 Laptop GPU
- CUDA version: 12.3.2
- cuDNN version installed: 9.0.0
- cuDNN version JAX expects: 9.1.1 (according to error messages)
- Container: Docker with NVIDIA CUDA 12.3.2 base image

### Environment Variables
- JAX_PLATFORM_NAME=gpu
- XLA_PYTHON_CLIENT_PREALLOCATE=false
- XLA_PYTHON_CLIENT_ALLOCATOR=platform
- XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

## Symptoms 
When running JAX operations in a Jupyter notebook, the kernel silently crashes and restarts without producing explicit Python exceptions. The issue has these characteristics:

1. Simple JAX import works but any jnp.dot() or other operations that use cuDNN crash
2. When running with PYTHONFAULTHANDLER=1, I see the following error in logs:
```
E0508 03:34:06.237713 14701 cuda_dnn.cc:522] Loaded runtime CuDNN library: 9.0.0 but source was compiled with: 9.1.1. CuDNN library needs to have matching major version and equal or higher minor version. If using a binary install, upgrade your CuDNN library. If building from sources, make sure the library loaded at runtime is compatible with the version specified during compile configuration.
```

3. The crash happens consistently with simple matrix operations or PyMC sampling using NumPyro backend

## Repro Steps

```python
import jax
import jax.numpy as jnp

# This works fine
print(jax.__version__)
print(jax.devices())

# This crashes the kernel silently
x = jnp.ones((1000, 1000))
y = jnp.dot(x, x)
y.block_until_ready()
```

## Diagnostic Information

I ran a complete diagnostic script that confirmed:
1. libcudnn.so.9 loads successfully via ctypes
2. JAX imports correctly
3. JAX detects the GPU correctly but fails when trying to execute operations
4. The cuDNN version mismatch is reported in XLA logs

<details>
<summary>Full diagnostic output</summary>

```
=== JAX DIAGNOSTICS REPORT (20250508_033709) ===
...
Found cuDNN libraries:
/usr/lib/x86_64-linux-gnu/libcudnn.so
/usr/lib/x86_64-linux-gnu/libcudnn.so.9
/usr/lib/x86_64-linux-gnu/libcudnn.so.9.0.0
...
E0508 03:34:06.237713 14701 cuda_dnn.cc:522] Loaded runtime CuDNN library: 9.0.0 but source was compiled with: 9.1.1.  CuDNN library needs to have matching major version and equal or higher minor version.
...
JAX operations test failed: FAILED_PRECONDITION: DNN library initialization failed
```
</details>

## Resolution Attempts

I've tried:
1. Setting XLA_PYTHON_CLIENT_PREALLOCATE=false (no effect)
2. Setting XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda" (no effect)
3. Upgrading/downgrading JAX versions (doesn't solve the core issue)

## Expected Solution

From the error message, it appears JAX 0.5.2 with jaxlib 0.5.1 was compiled against cuDNN 9.1.1 but my container has cuDNN 9.0.0 installed.

**Two suggested solutions for maintainers:**
1. Update JAX documentation to explicitly state which cuDNN versions are compatible with each JAX release
2. Consider handling version mismatches more gracefully with a Python-level exception that explains the incompatibility rather than a silent crash

**How I worked around it:**
I installed libcudnn9=9.1.0* and libcudnn9-dev=9.1.0* packages which resolved the issue.

## Questions for Maintainers
1. Is there an official compatibility matrix for JAX versions, CUDA versions, and cuDNN versions?
2. Could future JAX releases provide clearer Python errors for these issues? 