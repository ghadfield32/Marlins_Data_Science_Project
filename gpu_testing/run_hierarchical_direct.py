"""
Run hierarchical model with GPU support

location: \Marlins_Data_Science_Project\gpu_testing

"""
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Run the hierarchical model module directly
print("Running hierarchical model with JAX GPU support...")
from models.archive.hierarchical import fit_bayesian_hierarchical
from src.data.load_data import load_raw
from src.features.feature_engineering import feature_engineer
from src.features.preprocess import prepare_for_mixed_and_hierarchical

# Load and prepare data
raw_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
df = load_raw(raw_path)
print("Raw data loaded:", df.shape)

df_fe = feature_engineer(df)
print("Feature engineering complete:", df_fe.shape)

df_model = prepare_for_mixed_and_hierarchical(df_fe)
print("Model preparation complete:", df_model.shape)

# Extract arrays for PyMC
batter_idx = df_model["batter_id"].cat.codes.values
level_idx = df_model["level_idx"].values
age_centered = df_model["age_centered"].values
print("Arrays prepared for model")

# Fit the Bayesian hierarchical model
idata = fit_bayesian_hierarchical(
    df_model, batter_idx, level_idx, age_centered,
    mu_prior=90, sigma_prior=5,
    sampler="jax",  # Using JAX with GPU
    draws=1000, tune=1000
)

print("Model fitting complete!")
print(idata) 