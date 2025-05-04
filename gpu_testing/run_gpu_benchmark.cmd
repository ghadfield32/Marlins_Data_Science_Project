@echo off
REM Run a matrix multiplication benchmark to test JAX GPU performance
echo Running JAX matrix multiplication benchmark...

REM Check if jax-gpu-test image exists, build if not
docker image inspect jax-gpu-test >nul 2>&1
if %errorlevel% neq 0 (
    echo Building JAX GPU test container...
    docker build -t jax-gpu-test -f test_jax_gpu.dockerfile .
)

REM Run the matrix mult benchmark
echo.
echo Starting benchmark...
docker run --rm --gpus all -it -v "%cd%":/app -w /app jax-gpu-test python3 -c ^
"import jax; import jax.numpy as jnp; from jax import random, jit; import time; ^
print('JAX version:', jax.__version__); ^
print('JAX devices:', jax.devices()); ^
print('GPU count:', jax.device_count('gpu')); ^
key = random.key(0); ^
x = random.normal(key, (5000, 5000), dtype=jnp.float32); ^
@jit ^
def matmul(x): return jnp.dot(x, x.T); ^
_ = matmul(x).block_until_ready(); ^
start = time.time(); ^
result = matmul(x).block_until_ready(); ^
elapsed = time.time() - start; ^
print('Matrix multiplication time: %0.4f seconds' % elapsed); ^
print('GPU acceleration working:', elapsed < 0.5)"

echo.
echo Benchmark complete!
echo If matrix multiplication time is less than 0.5 seconds, GPU acceleration is working correctly. 