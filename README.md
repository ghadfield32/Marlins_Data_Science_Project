# Marlins Data Science Project

This repository contains the data science pipeline for the Marlins exit velocity project, now with GPU-accelerated modeling using JAX, PyMC, and NumPyro.

## Containerized GPU Setup

This project uses Docker with NVIDIA's container toolkit to provide GPU acceleration for JAX, PyMC, and NumPyro models. The setup includes:

- **Docker + CUDA 12.4.1**: Base image with NVIDIA GPU support
- **JupyterLab**: Interactive notebook environment for data exploration and model development
- **VS Code DevContainer**: Seamless development experience with GPU support
- **Invoke Tasks**: Python-based task automation for reproducible workflows
- **Parquet Data Caching**: Faster I/O by converting CSVs to Parquet format

### Requirements

- Docker and Docker Compose
- NVIDIA GPU with compatible drivers
- NVIDIA Container Toolkit installed
- Visual Studio Code with Remote Containers extension (optional, for devcontainer)

## Getting Started

### Using VS Code DevContainer (Recommended)

1. Install the VS Code Remote - Containers extension
2. Open this project folder in VS Code
3. VS Code will prompt you to "Reopen in Container" (or use the command palette)
4. The container will build and you'll have a fully configured environment with GPU support

### Using Docker Compose Directly

```bash
# Build and start the container
docker-compose up -d

# Run a JAX GPU test to verify GPU access
docker-compose exec marlins-jax python -c "import jax; print(jax.devices())"

# Open JupyterLab
docker-compose exec marlins-jax jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Then visit http://localhost:8888 in your browser.

## Using Invoke Tasks

This project uses [Invoke](https://www.pyinvoke.org/) instead of Makefiles for task automation:

```bash
# Verify GPU is working
invoke gpu-test

# Run the hierarchical model with GPU support
invoke hierarchical --draws 1000 --tune 1000

# Start JupyterLab
invoke jupyter

# Convert CSV to Parquet for faster loading
invoke convert-to-parquet --csv-path data/your_file.csv
```

## Project Structure

```
.
├── data/                  # Data directory
│   ├── models/            # Saved models
│   └── Research Data/     # Raw research data
├── src/                   # Source code
│   ├── data/              # Data loading and processing
│   ├── features/          # Feature engineering
│   ├── models/            # Model definitions and training
│   └── utils/             # Utility functions
├── gpu_testing/           # GPU-specific testing scripts
├── .devcontainer/         # VS Code devcontainer configuration
├── Dockerfile             # GPU-enabled container definition
├── docker-compose.yml     # Container orchestration
├── invoke.py              # Task automation
└── README.md              # This file
```

## Notes on GPU Acceleration

The container is configured to use JAX with GPU support. PyMC is configured to use JAX as the sampling backend via the `PYTENSOR_FLAGS="mode=JAX,floatX=float32"` environment variable. This provides significant speedup for MCMC sampling in hierarchical models.

To check if GPU is properly detected:

```python
import jax
print("GPU devices:", [d for d in jax.devices() if d.platform == "gpu"])
```

## License

[Your license information] 