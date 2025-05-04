@echo off
REM Run the hierarchical model with GPU support
echo Creating Python script to run hierarchical model...

REM Create a simple Python script to run the model
echo import os > run_hierarchical.py
echo import sys >> run_hierarchical.py
echo import jax >> run_hierarchical.py
echo. >> run_hierarchical.py
echo print('JAX version:', jax.__version__) >> run_hierarchical.py
echo print('JAX devices:', jax.devices()) >> run_hierarchical.py
echo print('GPU count:', jax.device_count('gpu')) >> run_hierarchical.py
echo jax.config.update('jax_enable_x64', True) >> run_hierarchical.py
echo. >> run_hierarchical.py
echo print('Loading data modules...') >> run_hierarchical.py
echo from src.data.load_data import load_raw >> run_hierarchical.py
echo from src.features.feature_engineering import feature_engineer >> run_hierarchical.py
echo from src.features.preprocess import prepare_for_mixed_and_hierarchical >> run_hierarchical.py
echo. >> run_hierarchical.py
echo print('Loading data...') >> run_hierarchical.py
echo raw_path = 'data/Research Data Project/Research Data Project/exit_velo_project_data.csv' >> run_hierarchical.py
echo df = load_raw(raw_path) >> run_hierarchical.py
echo print('Raw data loaded:', df.shape) >> run_hierarchical.py
echo df_fe = feature_engineer(df) >> run_hierarchical.py
echo print('Feature engineering complete:', df_fe.shape) >> run_hierarchical.py
echo df_model = prepare_for_mixed_and_hierarchical(df_fe) >> run_hierarchical.py
echo print('Model preparation complete:', df_model.shape) >> run_hierarchical.py
echo batter_idx = df_model['batter_id'].cat.codes.values >> run_hierarchical.py
echo level_idx = df_model['level_idx'].values >> run_hierarchical.py
echo age_centered = df_model['age_centered'].values >> run_hierarchical.py
echo print('Arrays prepared for model') >> run_hierarchical.py
echo. >> run_hierarchical.py
echo print('Running hierarchical model with GPU support...') >> run_hierarchical.py
echo from src.models.hierarchical import fit_bayesian_hierarchical >> run_hierarchical.py
echo idata = fit_bayesian_hierarchical( >> run_hierarchical.py
echo     df_model, batter_idx, level_idx, age_centered, >> run_hierarchical.py
echo     mu_prior=90, sigma_prior=5, >> run_hierarchical.py
echo     sampler='jax', >> run_hierarchical.py
echo     draws=1000, tune=1000 >> run_hierarchical.py
echo ) >> run_hierarchical.py
echo print('Model fitting complete!') >> run_hierarchical.py

echo Running Hierarchical Model with JAX GPU support...

REM Run the model using the Docker container with the Python script
docker run --rm --gpus all -it -v "%cd%\..":/app -w /app jax-gpu-test bash -c "pip install pandas pymc>=5.0.0 arviz scikit-learn tqdm shap matplotlib && cd /app && python3 gpu_testing/run_hierarchical.py"

echo.
echo Hierarchical model run complete! 