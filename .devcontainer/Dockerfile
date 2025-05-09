# .devcontainer/Dockerfile
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ARG ENV_NAME=marlins-ds-gpu
ARG PYTHON_VER=3.10
ARG JAX_PREALLOCATE=true
ARG JAX_MEM_FRAC=0.95
ARG JAX_ALLOCATOR=platform
ARG JAX_PREALLOC_LIMIT=8589934592

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
# make sure jaxlib's RPATH wins
ENV LD_LIBRARY_PATH=/opt/conda/envs/${ENV_NAME}/lib:$LD_LIBRARY_PATH

# 1. Install general system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      wget \
      bzip2 \
      ca-certificates \
      jags libjags-dev \
      apt-transport-https \
      gnupg2 \
      curl \
      git \
      vim \
      python3-pip \
      python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Install PyJAGS into the conda env via pip
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install --no-cache-dir pyjags

# ------------------------------------------------------------------------
# The base image already provides:
#   * CUDA 12.3 repo key + sources
#   * cuDNN 9.1.1 runtime & dev headers
# so no extra repository or libcudnn install is necessary.
# ------------------------------------------------------------------------

# 4. Miniconda installation
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    conda clean -a -y

# 5. Create the conda environment
COPY .devcontainer/environment.yml /tmp/environment.yml
RUN sed -i "s/^name: .*/name: $ENV_NAME/" /tmp/environment.yml && \
    sed -i "s/^- python=.*/- python=$PYTHON_VER/" /tmp/environment.yml && \
    conda env create -f /tmp/environment.yml && \
    conda clean -a -y

# 6. Ensure the environment auto-activates
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME" \
      >> /etc/bash.bashrc

# 7. Tune JAX GPU behavior - set all environment variables from the memory fix module
ENV XLA_PYTHON_CLIENT_PREALLOCATE=${JAX_PREALLOCATE}
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=${JAX_MEM_FRAC}
ENV XLA_PYTHON_CLIENT_ALLOCATOR=${JAX_ALLOCATOR}
ENV JAX_PLATFORM_NAME=gpu
ENV XLA_FLAGS="--xla_force_host_platform_device_count=1 --xla_gpu_deterministic_reductions"
ENV JAX_DISABLE_JIT=false
ENV JAX_ENABLE_X64=false
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV JAX_PREALLOCATION_SIZE_LIMIT_BYTES=${JAX_PREALLOC_LIMIT}

# 8. Install JAX CUDA-enabled wheel
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install --no-cache-dir --upgrade "jax[cuda12]==0.5.2" \
      -f https://storage.googleapis.com/jax-releases/jax_releases.html

# 9. Install IPython kernel for Jupyter
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate ${ENV_NAME} && \
    python -m ipykernel install --name ${ENV_NAME} \
      --display-name "Python (${ENV_NAME})"

WORKDIR /workspace
CMD ["/bin/bash"]

# 10. Deduplicate LD_LIBRARY_PATH on shell startup
RUN echo 'export LD_LIBRARY_PATH=$(printf "%s" "$LD_LIBRARY_PATH" \
    | tr ":" "\n" | awk "!seen[$0]++" | paste -sd ":")' \
    >> /etc/profile.d/clean_ld.sh
