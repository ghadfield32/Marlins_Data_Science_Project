@echo off
echo Running JAX with GPU support...

REM Check if jax-gpu-test image exists, build if not
docker image inspect jax-gpu-test >nul 2>&1
if %errorlevel% neq 0 (
    echo Building JAX GPU test container...
    docker build -t jax-gpu-test -f test_jax_gpu.dockerfile .
)

REM Run the GPU test in interactive mode
echo Running basic GPU test...
docker run --rm --gpus all -it -v "%cd%":/app -w /app jax-gpu-test python3 -c "import jax; print('JAX version:', jax.__version__); print('JAX Devices:', jax.devices()); print('GPU count:', jax.device_count('gpu')); print('Default backend:', jax.default_backend())"

echo.
echo To run your own JAX script with GPU support:
echo docker run --rm --gpus all -it -v "%%cd%%":/app -w /app jax-gpu-test python3 your_script.py
echo.
echo To run a Python REPL with JAX and GPU:
echo docker run --rm --gpus all -it -v "%%cd%%":/app -w /app jax-gpu-test python3
echo.
echo To run the matrix multiplication benchmark:
echo docker run --rm --gpus all -it -v "%%cd%%":/app -w /app jax-gpu-test python3 gpu_testing/gpu_test.py 