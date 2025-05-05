# .devcontainer/Dockerfile  ─────────────────────────────────────────────
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ARG ENV_NAME=marlins-ds-gpu
ARG PYTHON_VER=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# System deps
RUN apt-get update --fix-missing && apt-get install -y \
    wget bzip2 ca-certificates curl git vim python3-pip python3-dev \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
 && /bin/bash ~/miniconda.sh -b -p /opt/conda \
 && rm ~/miniconda.sh \
 && $CONDA_DIR/bin/conda clean -a -y

# Conda env
COPY .devcontainer/environment.yml /tmp/environment.yml
RUN sed -i "s/name: .*/name: $ENV_NAME/" /tmp/environment.yml \
 && sed -i "s/- python=.*/- python=$PYTHON_VER/" /tmp/environment.yml \
 && conda env create -f /tmp/environment.yml \
 && conda clean -a -y

# Activate on shell start
RUN echo ". /opt/conda/etc/profile.d/conda.sh && conda activate $ENV_NAME" >> /etc/bash.bashrc

# ── JAX + CUDA plugin ─────────────────────────────────────────────
RUN /bin/bash -c "\
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate $ENV_NAME && \
    pip install --upgrade pip && \
    # Preferred: pull GPU wheel via JAX extra
    pip install 'jax[cuda12]' \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
"




WORKDIR /workspace
CMD [\"/bin/bash\"]
