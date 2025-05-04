"""
Example script demonstrating PyMC with JAX backend for GPU acceleration.
"""
import os
import time
import sys
import numpy as np

# Configure environment variables for PyTensor JAX backend
os.environ["PYTENSOR_FLAGS"] = "mode=JAX,floatX=float32"
os.environ["JAX_PLATFORMS"] = "cpu,cuda"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

# Import JAX to check GPU availability
try:
    import jax
    import jaxlib
    
    # Check JAX version and backend
    print(f"JAX version: {jax.__version__}, jaxlib: {jaxlib.__version__}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    
    # Check for GPU
    has_jax_gpu = any(d.platform == 'gpu' for d in jax.devices())
    print(f"JAX has GPU access: {has_jax_gpu}")
    
    if not has_jax_gpu:
        print("\nWARNING: GPU not detected by JAX. Possible causes:")
        print("  - Using CPU-only JAX/jaxlib wheels")
        print("  - Missing CUDA drivers or incompatible CUDA version")
        print("  - Docker container not configured with GPU access")
        print("\nIf using Docker, make sure:")
        print("  - NVIDIA Container Toolkit is installed")
        print("  - Container is started with --gpus all flag")
        print("  - In docker-compose, the GPU device is properly configured")
        
        # Exit if we're expecting GPU but can't find it
        if os.environ.get("REQUIRE_GPU", "").lower() in ("1", "true", "yes"):
            print("\nERROR: GPU access required but not available. Exiting.")
            sys.exit(1)
    
except ImportError:
    print("JAX not found. Install with: pip install jax[cuda12_pip]")
    sys.exit(1)

# Import PyMC and related libraries
try:
    import pymc as pm
    import arviz as az
    import pytensor
    print(f"\nPyMC version: {pm.__version__}")
    print(f"PyTensor version: {pytensor.__version__}")
    print(f"PyTensor mode: {pytensor.config.mode}")
except ImportError:
    print("PyMC or PyTensor not found. Install with: pip install pymc")
    sys.exit(1)

# Generate some example data
print("\nGenerating sample data...")
np.random.seed(42)
n_groups = 8
n_per_group = 30
group_means = np.random.normal(0, 1, n_groups)
idx = np.repeat(np.arange(n_groups), n_per_group)
y = np.random.normal(group_means[idx], 1)

print(f"Data shape: {y.shape}")
print(f"Number of groups: {n_groups}")

# Define and run a hierarchical model
print("\nCreating hierarchical model...")
start_time = time.time()

with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu = pm.Normal("mu", mu=0, sigma=1)
    sigma = pm.HalfNormal("sigma", sigma=1)
    
    # Varying intercepts
    group_mean = pm.Normal("group_mean", mu=mu, sigma=sigma, shape=n_groups)
    
    # Likelihood
    y_obs = pm.Normal("y_obs", mu=group_mean[idx], sigma=1, observed=y)
    
    # Sample from the posterior
    print("Sampling from posterior...")
    sample_time_start = time.time()
    trace = pm.sample(
        draws=1000, 
        tune=1000,
        chains=2, 
        cores=1,  # Using 1 core since JAX handles parallelization differently
        target_accept=0.95,  # Increased from default 0.8 to reduce divergences
        return_inferencedata=True
    )
    sample_time_end = time.time()

total_time = time.time() - start_time
sample_time = sample_time_end - sample_time_start

print("\nSampling completed!")
print(f"Total runtime: {total_time:.2f} seconds")
print(f"Sampling time: {sample_time:.2f} seconds")

# Display summary statistics
print("\nPosterior Summary:")
summary = az.summary(trace, var_names=["mu", "sigma", "group_mean"])
print(summary.head(10))

# Check for divergences
n_divergent = trace.sample_stats.diverging.sum().item()
if n_divergent > 0:
    print(f"\nWarning: {n_divergent} divergent transitions detected.")
    print("Consider:")
    print("  - Increasing target_accept (currently 0.95)")
    print("  - Reparameterizing the model")
    print("  - Using a non-centered parameterization")

# Plot results
try:
    import matplotlib.pyplot as plt
    
    az.plot_trace(trace, var_names=["mu", "sigma"])
    plt.tight_layout()
    # plt.savefig("hierarchical_trace.png")
    print("\nTrace plot created (not saved)")
    
    az.plot_forest(trace, var_names=["group_mean"], combined=True)
    plt.tight_layout()
    # plt.savefig("hierarchical_forest.png")
    print("Forest plot created (not saved)")
except Exception as e:
    print(f"Error generating plots: {e}") 