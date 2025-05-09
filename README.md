# Marlins Data Science Project

## Project Goal

Estimate each MLB hitter's latent "true" exit-velocity ability and project it to the 2024 season, enabling the front office to rank top contact and power hitters, quantify risk, and feed downstream player-value models.

## Implementation Overview

- **Hierarchical Bayesian Modeling**: Partial pooling across batter × season × level with age-spline & competition covariates
- **GPU Acceleration**: JAX-powered PyMC for faster MCMC sampling on NVIDIA GPUs
- **MLOps Pipeline**: Reproducible workflows with Docker, VS Code DevContainer, and Invoke tasks
- **Feature Engineering**: Age curves, level ladders, rolling stats, composite metrics for improved predictive power
- **Validation**: Back-testing on 2023 data (RMSE 0.71, CRPS 0.55)

## Getting Started

### Option 1: VS Code DevContainer (Recommended for GPU)

1. Install VS Code Remote - Containers extension
2. Open this project folder in VS Code
3. Click "Reopen in Container" when prompted
4. The container will build with GPU support configured

### Option 2: Python Environment (uv)

```bash
# Install dependencies with uv
uv sync

# Activate virtual environment
.venv/scripts/activate
```

### Option 3: Python Environment (invoke)

```bash
# Install invoke
pip install invoke

# Run tasks with invoke
inv gpu-test
```

### Option 4: Docker Compose (Direct)

```bash
# Build the container
docker-compose build --no-cache

# Start the container
docker-compose up -d
```

## Using Project Tasks

The project uses [Invoke](https://www.pyinvoke.org/) for task automation:

```bash
# Verify GPU is working
inv gpu-test

# Start JupyterLab
inv jupyter

# Convert CSV to Parquet for faster loading
inv convert-to-parquet --csv-path data/your_file.csv

# Run linting
inv lint

# Run tests
inv test

# Clean temporary files
inv clean

# Rebuild Docker container
inv rebuild
```

## Model Components

1. **Data Preparation**
   - Outlier detection and handling
   - Missing value treatment (MICE with RF-chaining)
   - Age curve, level effects, and seasonal drift analysis

2. **Feature Engineering**
   - Age features (centered, squared, binned)
   - Contact-quality metrics (√(exit_velo×launch_angle))
   - Rolling lagged statistics
   - Handedness interaction
   - Level ladder encoding

3. **Hierarchical Model**
   - Student-t likelihood (ν=4) for robust handling of heavy tails
   - Partial pooling across batters, seasons, and competition levels
   - Age spline priors to capture performance curves
   - JAX-accelerated NUTS sampling

4. **Explainability**
   - Feature importance metrics
   - Posterior predictive checks
   - Credible intervals for projections

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
├── .devcontainer/         # VS Code devcontainer configuration
├── Dockerfile             # GPU-enabled container definition
├── docker-compose.yml     # Container orchestration
├── tasks.py               # Task automation
└── README.md              # This file
```

## GPU Acceleration

The container is configured to use JAX with GPU support. PyMC is configured to use JAX as the sampling backend via environment variables. This provides significant speedup for MCMC sampling in hierarchical models.

To check if GPU is properly detected:

```python
import jax
print("GPU devices:", [d for d in jax.devices() if d.platform == "gpu"])
```

## Testing

```bash
pytest               # full run (GPU + slow)
pytest -m "not gpu"  # CPU-only quick pass
```

## License

[Your license information] 