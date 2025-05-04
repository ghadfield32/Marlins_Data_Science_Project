"""
Simple PyMC example using JAX backend for GPU acceleration.
This is a minimal example to verify that PyMC is using JAX with GPU support.
"""
import os
import time
import numpy as np

# Configure environment variables for PyTensor JAX backend
os.environ["PYTENSOR_FLAGS"] = "mode=JAX,floatX=float32"
os.environ["JAX_PLATFORMS"] = "cpu,cuda"

# Print JAX GPU information
import jax
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"GPU count: {jax.device_count('gpu')}")
print(f"Default backend: {jax.default_backend()}")

# Import PyMC
import pymc as pm
import pytensor
print(f"PyMC version: {pm.__version__}")
print(f"PyTensor mode: {pytensor.config.mode}")
print(f"PyTensor floatX: {pytensor.config.floatX}")

# Simple linear regression
print("\nRunning simple linear regression with PyMC...")
start_time = time.time()

# Generate synthetic data
np.random.seed(42)
N = 1000
true_intercept = 1.0
true_slope = 2.0
x = np.random.normal(0, 1, N)
y = true_intercept + true_slope * x + np.random.normal(0, 0.5, N)

# Create and sample from model
with pm.Model() as model:
    # Priors
    intercept = pm.Normal("intercept", mu=0, sigma=10)
    slope = pm.Normal("slope", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=1)
    
    # Linear model
    mu = intercept + slope * x
    
    # Likelihood
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=y)
    
    # Sample
    print("Sampling posterior...")
    sample_start = time.time()
    trace = pm.sample(1000, tune=1000, chains=2, cores=1, return_inferencedata=True)
    sample_time = time.time() - sample_start

total_time = time.time() - start_time
print(f"\nSampling completed in {sample_time:.2f} seconds")
print(f"Total runtime: {total_time:.2f} seconds")

# Display results
print("\nPosterior summary:")
print(trace.posterior["intercept"].mean().item(), "(true: 1.0)")
print(trace.posterior["slope"].mean().item(), "(true: 2.0)")
print(trace.posterior["sigma"].mean().item(), "(true: 0.5)") 