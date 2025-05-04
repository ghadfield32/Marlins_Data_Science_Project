# PowerShell script to run PyMC with JAX GPU support
Write-Host "Setting up PyMC with JAX GPU support..." -ForegroundColor Green

# Check if jax-gpu-test image exists, build if not
try {
    docker image inspect jax-gpu-test | Out-Null
} catch {
    Write-Host "Building JAX GPU test container..." -ForegroundColor Yellow
    docker build -t jax-gpu-test -f test_jax_gpu.dockerfile .
}

# Install PyMC dependencies if needed
Write-Host "Installing PyMC and dependencies (if needed)..." -ForegroundColor Yellow
docker run --rm --gpus all -it jax-gpu-test pip install 'pymc>=5.0.0' arviz 'pytensor>=2.30.0'

# Run the simple PyMC example
Write-Host "`nRunning PyMC example with JAX GPU support..." -ForegroundColor Green
docker run --rm --gpus all -it -v "${PWD}:/app" -w /app jax-gpu-test python3 gpu_testing/pymc_simple_example.py

Write-Host "`nTo run your own PyMC script with JAX GPU support:" -ForegroundColor Cyan
Write-Host "docker run --rm --gpus all -it -v `"${PWD}:/app`" -w /app jax-gpu-test python3 your_script.py" -ForegroundColor White

Write-Host "`nTo run the full PyMC JAX example:" -ForegroundColor Cyan
Write-Host "docker run --rm --gpus all -it -v `"${PWD}:/app`" -w /app jax-gpu-test python3 gpu_testing/pymc_jax_example.py" -ForegroundColor White 