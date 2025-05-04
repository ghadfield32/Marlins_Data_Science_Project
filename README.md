# Marlins Baseball Data Science Project

## Project Overview
This project implements hierarchical Bayesian modeling for baseball exit velocity data, focusing on batter-specific effects and adjusting for covariates like player level and age.

## Docker Setup for JAX with GPU Support

This project uses Docker to provide a consistent environment with GPU support for JAX.

### Prerequisites

1. [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
2. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed
3. NVIDIA GPU drivers installed on your host machine

### Installation Steps for NVIDIA Container Toolkit on Windows

1. Install [NVIDIA GPU drivers](https://www.nvidia.com/Download/index.aspx) for your GPU
2. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)
3. Enable WSL 2 backend in Docker Desktop settings
4. Enable GPU support in Docker Desktop:
   - Go to Settings > Resources > WSL Integration
   - Enable "Use the WSL 2 based engine"
   - Go to Settings > General
   - Check "Use WSL 2 based engine"

### Building and Running the Container

```bash
# Build and start the container
docker compose up -d

# Run the JAX GPU test
docker compose exec marlins-jax python pymc_jax_example.py

# Alternatively, get a shell in the container
docker compose exec marlins-jax bash
```

### Checking GPU Access

To verify JAX has GPU access:

```python
import jax
print(jax.devices())  # Should show GPU devices
print(jax.default_backend())  # Should show 'gpu'
```

### Stopping the Container

```bash
docker compose down
```

### Troubleshooting

If you encounter GPU detection issues:

1. Verify NVIDIA drivers are properly installed:
   ```bash
   nvidia-smi
   ```

2. Check that Docker can see GPUs:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
   ```

3. Make sure you have the correct CUDA version in the container (12.x recommended for newer GPUs).

## Project Structure
```
Marlins_Data_Science_Project/
├─ data/                        # Raw data storage
│  └─ Research Data Project/    # Source data files
├─ src/                         # Source code
│  ├─ data/                     # Data loading
│  │   └─ load_data.py          # load_raw()
│  ├─ features/                 # Data processing
│  │   ├─ eda.py                # Exploratory data analysis
│  │   └─ preprocess.py         # Data cleaning and feature generation
│  ├─ models/                   # Statistical models
│  │   ├─ baseline.py           # Frequentist mixed models
│  │   └─ hierarchical.py       # Bayesian hierarchical models
│  └─ utils/                    # Utilities
│      ├─ helpers.py            # Common helper functions
│      └─ validation.py         # Model validation tools
```

## Environment Setup

### Using uv (recommended)
1. Install uv if not already installed: `pip install uv`
2. Create virtual environment and sync dependencies:
   ```
   # On Windows
   .\setup_env.bat
   
   # On Linux/macOS
   ./setup_env.sh
   ```
3. Activate the environment:
   ```
   # On Windows
   .\.venv\Scripts\activate
   
   # On Linux/macOS
   source .venv/bin/activate
   ```

### Manual setup
If you prefer to set up manually:
```
# Create environment
uv venv .venv

# Activate environment (Windows)
.\.venv\Scripts\activate

# Activate environment (Linux/macOS)
source .venv/bin/activate

# Install dependencies
uv pip sync
```

## Key Features
- **EDA & Visualization**: Tools for exploring data distributions and relationships
- **Hierarchical Modeling**: Bayesian models for batter skill estimation
- **Covariate Adjustment**: Incorporation of player level and age into models
- **Cross-validation**: Tools for model validation and comparison

## Requirements
- Python 3.7+
- PyMC3
- pandas
- numpy
- matplotlib
- scikit-learn
- statsmodels
- arviz

## Usage Examples
```python
# Load and preprocess data
from src.data.load_data import load_raw
from src.features.preprocess import preprocess

df_raw = load_raw()
df_clean = preprocess(df_raw)

# Exploratory analysis
from src.features.eda import plot_distributions, plot_correlations
plot_distributions(df_clean, by="level_abbr")
plot_correlations(df_clean, ["exit_velo", "launch_angle", "age"])

# Fit hierarchical model
from src.models.hierarchical import fit_bayesian_hierarchical
from src.utils.helpers import compute_league_prior

mu, sd = compute_league_prior(df_clean)
idata = fit_bayesian_hierarchical(
    df_clean,
    batter_idx=df_clean.batter_id.cat.codes.values,
    level_idx=df_clean.level_idx.values,
    age_centered=df_clean.age_centered.values,
    mu_prior=mu,
    sigma_prior=sd
)

# Validation
from src.utils.validation import posterior_predictive_check
posterior_predictive_check(idata, df_clean)
``` 