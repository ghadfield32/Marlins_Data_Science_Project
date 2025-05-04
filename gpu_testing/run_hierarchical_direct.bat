@echo off
REM Run the hierarchical model with direct Python execution
echo Running Hierarchical Model with JAX GPU support (direct method)...

REM Run the model in the Docker container with numpyro installed
docker run --rm --gpus all -it -v "%cd%\..":/app -w /app jax-gpu-test bash -c "pip install pandas pymc>=5.0.0 arviz scikit-learn tqdm shap matplotlib numpyro && cd /app && python3 gpu_testing/run_hierarchical_direct.py"

echo.
echo Hierarchical model run complete! 