#!/usr/bin/env python
"""
Test script for hierarchical model memory monitoring

This script tests the memory monitoring in the hierarchical bayesian model
to ensure it properly utilizes available GPU memory.
"""
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# First, import and log memory settings
from src.utils.jax_memory_fix_module import apply_jax_memory_fix
settings = apply_jax_memory_fix(fraction=0.9, preallocate=True, verbose=True)

# Log GPU diagnostics
from src.utils.jax_gpu_utils import log_gpu_diagnostics
log_gpu_diagnostics()

# Import hierarchical modeling functions
from src.models.hierarchical import fit_bayesian_hierarchical
from src.data.load_data import load_and_clean_data
from src.features.feature_engineering import feature_engineer
from src.features.preprocess import (
    fit_preprocessor,
    prepare_for_mixed_and_hierarchical
)

def create_synthetic_data(n_samples=500):
    """Create synthetic data suitable for hierarchical model."""
    np.random.seed(42)
    
    # Create batters, pitchers and levels
    n_batters = 30
    n_pitchers = 20
    levels = ['A', 'AA', 'AAA']
    seasons = [2021, 2022, 2023]
    
    # Generate random data
    df = pd.DataFrame({
        'batter_id': [f"BAT{i:03d}" for i in np.random.choice(n_batters, n_samples)],
        'pitcher_id': [f"PIT{i:03d}" for i in np.random.choice(n_pitchers, n_samples)],
        'exit_velo': 90 + np.random.randn(n_samples) * 5,
        'level_abbr': np.random.choice(levels, n_samples),
        'season': np.random.choice(seasons, n_samples),
        'age': 20 + np.random.randint(0, 10, n_samples),
        'launch_angle': 15 + np.random.randn(n_samples) * 10,
        'hit_type': np.random.choice(['GB', 'LD', 'FB'], n_samples),
        'batter_hand': np.random.choice(['L', 'R'], n_samples),
        'pitcher_hand': np.random.choice(['L', 'R'], n_samples),
        'batter_height': 70 + np.random.randint(0, 10, n_samples),
        'pitch_group': np.random.choice(['FF', 'SL', 'CH'], n_samples),
        'outcome': np.random.choice(['out', 'hit'], n_samples),
        'spray_angle': np.random.randn(n_samples) * 15,
        'hangtime': np.random.rand(n_samples) * 5,
    })
    
    print(f"Created synthetic dataset with {len(df)} rows")
    return df

def prepare_model_data(df):
    """Prepare data for hierarchical model."""
    # Create categorical indices
    batter_codes = pd.Categorical(df['batter_id']).codes
    level_codes = pd.Categorical(df['level_abbr']).codes
    season_codes = pd.Categorical(df['season'].astype(str)).codes
    pitcher_codes = pd.Categorical(df['pitcher_id']).codes
    
    # Add these as columns
    df_model = df.copy()
    df_model['batter_idx'] = batter_codes
    df_model['level_idx'] = level_codes
    df_model['season_idx'] = season_codes
    df_model['pitcher_idx'] = pitcher_codes
    
    print(f"Prepared model data with indices - Batters: {batter_codes.max()+1}, Levels: {level_codes.max()+1}, Seasons: {season_codes.max()+1}, Pitchers: {pitcher_codes.max()+1}")
    return df_model, batter_codes, level_codes, season_codes, pitcher_codes

def run_small_hierarchical_test():
    """Run a small test of the hierarchical model with memory monitoring."""
    print("\n=== Running Hierarchical Model Memory Test ===\n")
    
    # Create synthetic data
    df = create_synthetic_data(n_samples=500)
    
    # Prepare data with indices
    df_model, b_idx, l_idx, s_idx, p_idx = prepare_model_data(df)
    
    # Create preprocessor
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    
    # Define features to use
    cat_features = ['hit_type', 'batter_hand', 'pitcher_hand', 'pitch_group', 'outcome']
    num_features = ['age', 'batter_height', 'launch_angle', 'spray_angle']
    
    # Define preprocessor
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )
    
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop='first', sparse_output=False))
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)
        ]
    )
    
    # Fit the preprocessor
    X = df_model[num_features + cat_features]
    y = df_model['exit_velo']
    
    # Fit preprocessor
    X_transformed = preprocessor.fit_transform(X)
    print(f"Transformed features shape: {X_transformed.shape}")
    
    # Set small values for faster test
    draws_and_tune = 50
    chains = 2
    
    # Fit model with memory monitoring
    print("\nFitting hierarchical model with memory monitoring...\n")
    
    # Make sure arrays are numpy arrays
    b_idx = np.array(b_idx)
    l_idx = np.array(l_idx)
    s_idx = np.array(s_idx)
    p_idx = np.array(p_idx)
    
    # Get feature names
    feature_names = []
    for name, _, column in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(num_features)
        elif name == 'cat':
            # For each categorical feature, get all one-hot encoded column names
            for cat_feature in column:
                # Get categories (excluding the first one that's dropped)
                categories = df_model[cat_feature].unique()[1:]
                feature_names.extend([f"{cat_feature}_{cat}" for cat in categories])
    
    # Use direct feature input to avoid preprocessing issues
    idata = fit_bayesian_hierarchical(
        df_model,
        preprocessor,
        b_idx,            # batter codes
        l_idx,            # level codes
        s_idx,            # season codes
        p_idx,            # pitcher codes
        sampler="jax",
        draws=draws_and_tune,
        tune=draws_and_tune,
        target_accept=0.95,
        chains=chains,
        monitor_memory=True,
        force_memory_allocation=True,
        allocation_target=0.8,  # Target 80% utilization
        direct_feature_input=(X_transformed, df_model['exit_velo'].values, feature_names)
    )
    
    print("\nTest complete!")
    return idata

if __name__ == "__main__":
    run_small_hierarchical_test() 