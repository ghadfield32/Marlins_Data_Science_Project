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

# 1. Install system deps (added build tools for PyJAGS)
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      software-properties-common && \
    add-apt-repository universe && \
    apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
      wget \
      bzip2 \
      ca-certificates \
      apt-transport-https \
      gnupg2 \
      curl \
      git \
      vim \
      python3-pip \
      python3-dev \
      pkg-config \
      build-essential \
      autoconf \
      automake \
      libtool \
      m4 \
      cmake && \
    rm -rf /var/lib/apt/lists/*



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

# 6. Install PyJAGS with cstdint header fix
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate ${ENV_NAME} && \
    # Create a simple patch script
    echo '#!/usr/bin/env python3\nimport os\nos.environ["CPPFLAGS"] = "-include cstdint"\nprint("Installing PyJAGS with cstdint header fix...")\nos.system("CC=x86_64-conda-linux-gnu-cc CXX=x86_64-conda-linux-gnu-c++ pip install --no-build-isolation pyjags==1.3.8")\nprint("PyJAGS installation complete!")' > /tmp/install_pyjags.py && \
    # Run the install script
    python /tmp/install_pyjags.py && \
    # Verify installation
    python -c "import sys; print('Attempting to import PyJAGS...')" && \
    python -c "import pyjags; print('PyJAGS OK →', pyjags.__version__)" || \
    # Fallback: try conda install if pip fails
    (conda install -c conda-forge -y pyjags=1.3 && \
    python -c "import pyjags; print('PyJAGS OK (conda) →', pyjags.__version__)") && \
    # Clean up
    rm -f /tmp/install_pyjags.py

# 7. Ensure the environment auto-activates
RUN echo ". $CONDA_DIR/etc/profile.d/conda.sh && conda activate $ENV_NAME" \
      >> /etc/bash.bashrc

# 8. Tune JAX GPU behavior - set all environment variables from the memory fix module
ENV XLA_PYTHON_CLIENT_PREALLOCATE=${JAX_PREALLOCATE}
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=${JAX_MEM_FRAC}
ENV XLA_PYTHON_CLIENT_ALLOCATOR=${JAX_ALLOCATOR}
ENV JAX_PLATFORM_NAME=gpu
ENV XLA_FLAGS="--xla_force_host_platform_device_count=1"
ENV JAX_DISABLE_JIT=false
ENV JAX_ENABLE_X64=false
ENV TF_FORCE_GPU_ALLOW_GROWTH=false
ENV JAX_PREALLOCATION_SIZE_LIMIT_BYTES=${JAX_PREALLOC_LIMIT}

# 9. Install JAX CUDA-enabled wheel
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install --no-cache-dir --upgrade "jax[cuda12]==0.5.2" \
      -f https://storage.googleapis.com/jax-releases/jax_releases.html

# 10. Install IPython kernel for Jupyter
RUN . $CONDA_DIR/etc/profile.d/conda.sh && \
    conda activate ${ENV_NAME} && \
    python -m ipykernel install --name ${ENV_NAME} \
      --display-name "Python (${ENV_NAME})"

WORKDIR /workspace
CMD ["/bin/bash"]

# 11. Deduplicate LD_LIBRARY_PATH on shell startup
RUN echo 'export LD_LIBRARY_PATH=$(printf "%s" "$LD_LIBRARY_PATH" \
    | tr ":" "\n" | awk "!seen[$0]++" | paste -sd ":")' \
    >> /etc/profile.d/clean_ld.sh