"""

location: \Marlins_Data_Science_Project\gpu_testing\run_hierarchical.py

"""


import os
import sys

# Add the project root to the Python path
sys.path.append('/app')

import jax

print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
print('GPU count:', jax.device_count('gpu'))
jax.config.update('jax_enable_x64', True)

print('Loading data modules...')
try:
    from src.data.load_data import load_raw
    from src.features.feature_engineering import feature_engineer
    from src.features.preprocess import prepare_for_mixed_and_hierarchical
    print('Modules loaded successfully')
except Exception as e:
    print('Error loading modules:', e)
    sys.exit(1)

print('Loading data...')
try:
    raw_path = 'data/Research Data Project/Research Data Project/exit_velo_project_data.csv'
    df = load_raw(raw_path)
    print('Raw data loaded:', df.shape)
    df_fe = feature_engineer(df)
    print('Feature engineering complete:', df_fe.shape)
    df_model = prepare_for_mixed_and_hierarchical(df_fe)
    print('Model preparation complete:', df_model.shape)
    batter_idx = df_model['batter_id'].cat.codes.values
    level_idx = df_model['level_idx'].values
    age_centered = df_model['age_centered'].values
    print('Arrays prepared for model')
except Exception as e:
    print('Error preparing data:', e)
    sys.exit(1)

print('Running hierarchical model with GPU support...')
try:
    from src.models.hierarchical import fit_bayesian_hierarchical
    idata = fit_bayesian_hierarchical(
        df_model, batter_idx, level_idx, age_centered,
        mu_prior=90, sigma_prior=5,
        sampler='jax',
        draws=1000, tune=1000
    )
    print('Model fitting complete!')
except Exception as e:
    print('Error fitting model:', e)
    sys.exit(1)
