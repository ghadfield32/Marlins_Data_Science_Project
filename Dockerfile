FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Set up environment variables for CUDA and JAX
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTENSOR_FLAGS="mode=JAX,floatX=float32" \
    JAX_PLATFORMS="cpu,cuda" \
    XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1" \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install JAX with CUDA support directly (no virtualenv)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir "jax[cuda12]>=0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    && python3 -c "import jax; print('JAX GPU available:', len([d for d in jax.devices() if d.platform == 'gpu']) > 0)"

# Install other data science libraries
RUN pip install --no-cache-dir numpy pandas matplotlib scikit-learn xgboost catboost lightgbm pymc arviz ipython jupyter

# Create directories for mounting volumes
RUN mkdir -p /app/data

# Copy project files
COPY . /app/
WORKDIR /app

# Set default command
CMD ["python3", "gpu_testing/gpu_test.py"] 