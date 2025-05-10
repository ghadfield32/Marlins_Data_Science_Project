"""
Utility functions for saving, loading, and working with gradient boosting models.
"""
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

def save_pipeline(model, preprocessor, path: str = "models/gbm_pipeline.joblib", extra_attrs=None):
    """
    Save both model and preprocessor together to a single file.
    
    Parameters:
    -----------
    model : estimator
        The trained model to save
    preprocessor : transformer
        The fitted preprocessor to save
    path : str
        The file path where the pipeline will be saved
    extra_attrs : dict, optional
        Additional attributes to save with the pipeline
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the save dictionary
    save_dict = {"model": model, "preprocessor": preprocessor}
    
    # Add any extra attributes
    if extra_attrs:
        save_dict.update(extra_attrs)
    
    # Save to disk
    joblib.dump(save_dict, out_path)
    print(f"✅ Pipeline saved to {out_path.resolve()}")

def load_pipeline(path: str = "models/gbm_pipeline.joblib"):
    """
    Load the model+preprocessor dict back into memory.
    Returns a tuple: (model, preprocessor)
    
    Parameters:
    -----------
    path : str
        The file path where the pipeline is saved
        
    Returns:
    --------
    tuple
        (model, preprocessor) pair
    """
    saved = joblib.load(path)
    return saved["model"], saved["preprocessor"]

def get_feature_importance(model, preprocessor=None, feature_names=None):
    """
    Extract feature importance from a model.
    
    Parameters:
    -----------
    model : estimator
        The trained model (supports XGBoost, LightGBM, CatBoost, and sklearn models)
    preprocessor : transformer, optional
        The preprocessor used to transform the data
    feature_names : list of str, optional
        The feature names to use (if not provided, will try to get from preprocessor)
        
    Returns:
    --------
    pd.Series
        A series with feature names as index and importance as values, sorted descending
    """
    # Try to extract the actual model from a pipeline
    if hasattr(model, 'steps'):
        model = model.steps[-1][1]  # Get the last step in the pipeline
    
    # Get feature names if not provided
    if feature_names is None and preprocessor is not None:
        try:
            feature_names = preprocessor.get_feature_names_out()
        except (AttributeError, ValueError):
            # Fall back to generic names if preprocessor doesn't support get_feature_names_out
            feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]
    
    # Different model types store feature importance differently
    try:
        if hasattr(model, 'feature_importances_'):
            # Most sklearn models
            importances = model.feature_importances_
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'get_score'):
            # XGBoost
            importances_dict = model.get_booster().get_score(importance_type='gain')
            if feature_names is not None:
                # Map feature indices (f0, f1, etc.) to names
                # This handles the case where XGBoost returns feature indices instead of names
                mapping = {f'f{i}': name for i, name in enumerate(feature_names)}
                importances_dict = {mapping.get(k, k): v for k, v in importances_dict.items()}
            importances = pd.Series(importances_dict).values
            feature_names = pd.Series(importances_dict).index.tolist()
        elif hasattr(model, 'feature_importance'):
            # LightGBM and CatBoost
            importances = model.feature_importance()
        else:
            # No recognized importance method
            return pd.Series(dtype=float)
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        return pd.Series(dtype=float)
    
    # Ensure feature_names match the length of importances
    if feature_names is not None and len(feature_names) != len(importances):
        print(f"Warning: feature_names length ({len(feature_names)}) doesn't match "
              f"importances length ({len(importances)}). Using generic names.")
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    # Create Series and sort by importance
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]
    
    importance_series = pd.Series(importances, index=feature_names)
    return importance_series.sort_values(ascending=False)

def get_model_metadata(model):
    """
    Extract metadata from a trained model.
    
    Parameters:
    -----------
    model : estimator
        The trained model
        
    Returns:
    --------
    dict
        Dictionary with model metadata
    """
    # Try to extract the actual model from a pipeline
    if hasattr(model, 'steps'):
        for name, estimator in reversed(model.steps):  # Start from the last step
            if hasattr(estimator, 'predict'):
                model_meta = {'pipeline': True, 'final_step': name}
                model = estimator
                break
    else:
        model_meta = {'pipeline': False}
    
    # Get model type and parameters
    model_meta['type'] = type(model).__name__
    
    # Get params depending on model type
    try:
        if hasattr(model, 'get_params'):
            model_meta['params'] = model.get_params()
        
        # Get additional info for tree-based models
        if hasattr(model, 'n_estimators'):
            model_meta['n_estimators'] = model.n_estimators
        
        if hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'trees_to_dataframe'):
            # For XGBoost, get tree count
            model_meta['tree_count'] = len(model.get_booster().get_dump())
    except Exception as e:
        model_meta['error'] = str(e)
    
    return model_meta

