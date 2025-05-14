# Marlins Senior Data Science Hackathon - Baseball Modeling

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/YourOrg/YourRepo/actions)
[![Coverage Status](https://img.shields.io/badge/coverage-95%25-blue)](https://coveralls.io/github/YourOrg/YourRepo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Marlins Senior Data Science Hackathon - Baseball Modeling](#marlins-senior-data-science-hackathon---baseball-modeling)
  - [Table of Contents](#table-of-contents)
  - [Project Goal](#project-goal)
  - [Implementation Overview](#implementation-overview)
  - [Getting Started](#getting-started)
    - [Option 1: VS Code DevContainer (Recommended for GPU)](#option-1-vs-code-devcontainer-recommended-for-gpu)
    - [Option 2: Python Environment (uv)](#option-2-python-environment-uv)
    - [Option 3: Docker Compose (Direct)](#option-3-docker-compose-direct)
  - [Usage](#usage)
  - [Using Project Tasks](#using-project-tasks)
  - [Model Explainer Dashboard](#model-explainer-dashboard)
  - [Top Features](#top-features)
  - [Hierarchical Model Results](#hierarchical-model-results)
  - [Best Performing Model](#best-performing-model)
  - [Model Components](#model-components)
  - [Project Structure](#project-structure)
  - [GPU Acceleration](#gpu-acceleration)
  - [Testing](#testing)
  - [Contributing](#contributing)
  - [License](#license)

## Project Goal

Estimate each MLB hitter's latent "true" exit-velocity ability and project it to the 2024 season. This enables the front office to:

* Rank top contact and power hitters
* Quantify prediction risk
* Integrate outputs into downstream player-value models

## Implementation Overview

* **Hierarchical Bayesian Modeling**: Partial pooling across batter × season × level with age-spline & competition covariates.
* **GPU Acceleration**: JAX-powered PyMC for faster MCMC sampling on NVIDIA GPUs.
* **MLOps Pipeline**: Reproducible workflows with Docker, VS Code DevContainer, and Invoke tasks.
* **Feature Engineering**: Age curves, level ladders, rolling stats, composite metrics for improved predictive power.

## Getting Started

### Option 1: VS Code DevContainer (Recommended for GPU)

1. Install the VS Code Remote - Containers extension.
2. Open this project folder in VS Code.
3. Click "Rebuild and Reopen without Cache in Container" when prompted.
4. The container will build with GPU support configured.

### Option 2: Python Environment (uv)

```bash
# Install dependencies with uv
uv sync

# Activate virtual environment
.venv/scripts/activate
```

### Option 3: Docker Compose (Direct)

```bash
# Build the container
docker-compose build --no-cache

# Start the container
docker-compose up -d
```

## Usage

After setup, you can launch the explainer dashboard and generate predictions for specific batters and seasons. For example:

```bash
# Launch dashboard on port 8050 with a sample of 500 rows
inv explainerdash --port 8050 --sample-size 500
```

Once running, open your browser to:

```
http://localhost:8050/?batter_id=101&season=2024
```

This URL filters the dashboard to show projections for batter ID 101 in the 2024 season. Adjust `batter_id` and `season` parameters as needed.

## Using Project Tasks

The project uses [Invoke](https://www.pyinvoke.org/) for task automation:

```bash
# Launch the model explainer dashboard (port, debug)
inv explainerdash --port 8050 --debug

# Run linting
inv lint

# Run tests
inv test

# Clean temporary files
inv clean

# Rebuild Docker container
inv rebuild
```

## Model Explainer Dashboard

```bash
# Launch the model explainer dashboard
inv explainerdash
```

Options:

* `--host`: Specify host (default: `127.0.0.1`)
* `--port`: Specify port (default: `8050`)
* `--debug`: Enable debug mode
* `--sample-size`: Number of samples from the dataset (default: `200`)

![Model Explainer Dashboard](https://example.com/dashboard.png)

## Top Features

* `hang_time` (high importance)
* `launch_angle`

## Hierarchical Model Results

**Top 5 power hitters (2024 projections):**

| batter\_id | hitter\_type | pred\_mean | pred\_lo95 | pred\_hi95 |
| ---------- | ------------ | ---------- | ---------- | ---------- |
| 380        | POWER        | 94.83      | 93.62      | 95.88      |
| 2106       | POWER        | 93.09      | 92.29      | 93.77      |
| 991        | POWER        | 92.95      | 90.95      | 94.73      |
| 101        | POWER        | 92.25      | 90.91      | 93.48      |
| 2225       | POWER        | 92.08      | 91.05      | 93.09      |

**Top 5 contact hitters (2024 projections):**

| batter\_id | hitter\_type | pred\_mean | pred\_lo95 | pred\_hi95 |
| ---------- | ------------ | ---------- | ---------- | ---------- |
| 970        | CONTACT      | 95.12      | 93.13      | 97.06      |
| 2074       | CONTACT      | 94.24      | 92.68      | 96.06      |
| 2139       | CONTACT      | 93.94      | 93.31      | 94.72      |
| 1556       | CONTACT      | 93.76      | 91.94      | 95.60      |
| 2499       | CONTACT      | 93.48      | 92.54      | 94.42      |

| Model                              | Hyperparameters                                                                                                | RMSE       | Explainability Tools                         |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------- | -------------------------------------------- |
| Ridge Regression (Linear Baseline) | α=1.0; random\_state=0                                                                                         | 12.2698    | statsmodels CIs; permutation importance      |
| XGBoost GBM                        | n\_estimators=335; learning\_rate=0.06318; max\_depth=6; subsample=0.7943; colsample\_bytree=0.9826            | **7.8807** | TreeSHAP; SmartExplainer; ExplainerDashboard |
| Mixed-Effects Model (Hierarchical) | random intercepts (batter, season, pitcher); ML estimation; optimizer=L-BFGS (maxiter=100; tol=1e-6)           | 13.2382    | statsmodels random-effects summaries         |
| PyMC-HMC (NUTS)                    | draws=500; tune=500; chains=4; random\_seed=42; progressbar=False                                              | 12.70      | ArviZ summaries; PPC plots                   |
| PyMC-ADVI (VI)                     | method='advi'; n=2500; sample draws=250                                                                        | 85.71      | ArviZ summaries; PPC plots                   |
| Stan (CmdStanPy HMC)               | iter\_sampling=500; iter\_warmup=500; chains=4; seed=42; force\_compile=True; generated quantities→ y\_obs     | 12.48      | ArviZ summaries; PPC plots                   |
| JAGS (PyJAGS Gibbs)                | adapt=500; sample=500; chains=4; RNGs rotated; init seeds=42+10×chain                                          | 12.70      | ArviZ summaries; PPC plots                   |
| NumPyro (NUTS)                     | fraction=0.10; preallocate=False; draws=500; warmup=500; chains=4; progress\_bar=False; RNGKey=42; x64 enabled | 12.70      | ArviZ summaries; PPC plots                   |
| TFP-HMC (HMC)                      | step\_size=0.05; leapfrog\_steps=5; adaptation\_steps=400; draws=500; burnin=500; seed=42                      | 90.50      | ArviZ summaries; PPC plots                   |


## Best Performing Model

XGBoost GBM achieved the lowest RMSE of 7.8807, outperforming all other approaches. Its gradient-boosted decision trees, coupled with optimal hyperparameter tuning, deliver superior predictive accuracy for exit-velocity projections while still supporting robust explainability via TreeSHAP and the ExplainerDashboard.

## Model Components

1. **Data Preparation**

   * Outlier detection and handling
   * Missing value treatment (MICE with RF-chaining)
   * Age curve, level effects, and seasonal drift analysis
2. **Feature Engineering**

   * Age features (centered, squared, binned)
   * Contact-quality metrics (√(exit\_velo×launch\_angle))
   * Rolling lagged statistics
   * Handedness interaction
   * Level ladder encoding
3. **Hierarchical Model**

   * Student-t likelihood (ν=4) for heavy tails
   * Partial pooling across batters, seasons, levels
   * Age spline priors
   * JAX-accelerated NUTS sampling
4. **Explainability**

   * Feature importance metrics
   * Posterior predictive checks
   * Credible intervals
   * Interactive dashboard (explainerdashboard)

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
├── .devcontainer/         # DevContainer configuration
├── Dockerfile             # GPU-enabled container definition
├── docker-compose.yml     # Container orchestration
├── tasks.py               # Task automation
└── README.md              # This file
```

## GPU Acceleration

The container uses JAX with GPU support. PyMC is configured to use JAX as the sampling backend via environment variables for faster MCMC sampling.

```python
import jax
print("GPU devices:", [d for d in jax.devices() if d.platform == "gpu"])
```

## Testing

```bash
pytest               # full run (GPU + slow)
pytest -m "not gpu"  # CPU-only quick pass
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Code Style**: Adhere to PEP8. Use `black` for formatting and `flake8` for linting.
2. **Branch Workflow**: Fork the repo and create topic branches (`feature/`, `bugfix/`, `docs/`). Submit PRs against `main`.
3. **Issue Templates**: Use the provided issue templates in `.github/ISSUE_TEMPLATE/`.
4. **Pull Requests**: Include tests and update documentation. Ensure CI passes before requesting review.

## License

MIT License
