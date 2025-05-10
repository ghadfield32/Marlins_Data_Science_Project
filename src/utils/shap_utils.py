
"""
Utilities for safely handling SHAP value calculations.
"""
import numpy as np
import pandas as pd
import warnings
from functools import wraps
from pathlib import Path
import joblib

def safe_shap_values(model, X, feature_names=None, max_features=100, sample_size=None):
    """
    Safely calculate SHAP values for a model, handling common issues:
    1. Too many features causing memory or dimension mismatch errors
    2. NaN values in data
    3. Different model types requiring different explainers
    4. XGBoost dimension mismatch errors
    
    Parameters:
    -----------
    model : estimator
        The trained model
    X : pandas DataFrame or numpy array
        The input data for SHAP value calculation
    feature_names : list, optional
        Feature names to use. If None, will use X.columns if X is a DataFrame
    max_features : int, default=100
        Maximum number of features to include in SHAP calculation
    sample_size : int, optional
        If provided, will sample this many rows from X to reduce computation time
        
    Returns:
    --------
    tuple
        (shap_values, expected_value) or (None, None) if SHAP values couldn't be calculated
    """
    import shap
    
    # Convert X to DataFrame if it's not already
    if not isinstance(X, pd.DataFrame):
        if feature_names is not None:
            X = pd.DataFrame(X, columns=feature_names)
        else:
            X = pd.DataFrame(X)
    
    # Handle missing values
    if X.isna().any().any():
        X = X.fillna(X.mean())
    
    # Sample if requested
    if sample_size is not None and len(X) > sample_size:
        X = X.sample(sample_size, random_state=42)
    
    # Get feature names
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    # Limit features if there are too many
    if len(feature_names) > max_features:
        # Try to get feature importance if available
        if hasattr(model, 'feature_importances_') and len(model.feature_importances_) == len(feature_names):
            importance = pd.Series(model.feature_importances_, index=feature_names)
            top_features = importance.sort_values(ascending=False).head(max_features).index.tolist()
        elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'get_score'):
            # XGBoost models
            importance_dict = model.get_booster().get_score(importance_type='gain')
            # Map feature indices to names if needed
            if all(k.startswith('f') for k in importance_dict.keys()):
                importance_dict = {feature_names[int(k[1:])]: v for k, v in importance_dict.items() 
                                   if k[1:].isdigit() and int(k[1:]) < len(feature_names)}
            importance = pd.Series(importance_dict)
            top_features = importance.sort_values(ascending=False).head(max_features).index.tolist()
        else:
            # No feature importance available, just take the first max_features
            top_features = feature_names[:max_features]
        
        X = X[top_features]
        feature_names = top_features
    
    # Try to extract actual model from pipeline
    actual_model = model
    if hasattr(model, 'steps'):
        actual_model = model.steps[-1][1]
    
    # Determine the right explainer to use
    try:
        if hasattr(actual_model, 'get_booster'):
            # XGBoost model - try different approaches to handle dimension mismatches
            try:
                # First attempt: Default behavior (normal tree_limit)
                explainer = shap.TreeExplainer(actual_model, X)
                shap_values = explainer.shap_values(X)
                return shap_values, explainer.expected_value
            except Exception as e1:
                try:
                    # Second attempt: Manually set a tree_limit
                    n_trees = len(actual_model.get_booster().get_dump())
                    tree_limit = min(100, n_trees)  # Limit to at most 100 trees
                    explainer = shap.TreeExplainer(actual_model, X)
                    shap_values = explainer.shap_values(X, tree_limit=tree_limit)
                    return shap_values, explainer.expected_value
                except Exception as e2:
                    try:
                        # Third attempt: Use the original_model approach
                        class ModelWrapper:
                            def __init__(self, model):
                                self.original_model = model
                                self.model_output = "raw"

                        explainer = shap.TreeExplainer(ModelWrapper(actual_model))
                        # Use model.predict instead of shap_values directly
                        shap_values = explainer(X)
                        return shap_values.values, shap_values.base_values
                    except Exception as e3:
                        print(f"All XGBoost SHAP attempts failed:")
                        print(f"Attempt 1: {e1}")
                        print(f"Attempt 2: {e2}")
                        print(f"Attempt 3: {e3}")
                        return None, None
                        
        elif hasattr(actual_model, 'predict'):
            # Generic ML model - use Kernel explainer as fallback
            if hasattr(actual_model, 'predict_proba'):
                # For classifiers
                explainer = shap.KernelExplainer(
                    actual_model.predict_proba, shap.kmeans(X, min(20, len(X)))
                )
            else:
                # For regressors
                explainer = shap.KernelExplainer(
                    actual_model.predict, shap.kmeans(X, min(20, len(X)))
                )
            shap_values = explainer.shap_values(X, nsamples=100)
            return shap_values, explainer.expected_value
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        return None, None

def create_shap_summary_plot(model, X, feature_names=None, max_features=10, plot_type='bar', 
                             class_names=None, show=True, save_path=None):
    """
    Create and optionally save a SHAP summary plot
    
    Parameters:
    -----------
    model : estimator
        The trained model
    X : pandas DataFrame or numpy array
        The input data for SHAP value calculation
    feature_names : list, optional
        Feature names to use. If None, will use X.columns if X is a DataFrame
    max_features : int, default=10
        Maximum number of features to show in the plot
    plot_type : str, default='bar'
        Type of plot ('bar', 'violin', or 'dot')
    class_names : list, optional
        For classification, the class names to display
    show : bool, default=True
        Whether to display the plot
    save_path : str, optional
        If provided, will save the plot to this path
        
    Returns:
    --------
    tuple
        (shap_values, expected_value) that were used to generate the plot
    """
    import shap
    import matplotlib.pyplot as plt
    
    # Calculate SHAP values safely
    shap_values, expected_value = safe_shap_values(model, X, feature_names, max_features)
    
    if shap_values is None:
        print("Could not calculate SHAP values")
        return None, None
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Handle plot type
    if plot_type == 'bar':
        shap.summary_plot(shap_values, X, plot_type='bar', show=False, max_display=max_features)
    elif plot_type == 'violin':
        shap.summary_plot(shap_values, X, show=False, max_display=max_features)
    elif plot_type == 'dot':
        shap.summary_plot(shap_values, X, plot_type='dot', show=False, max_display=max_features)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()
    
    return shap_values, expected_value

def shap_feature_dependence_plot(model, X, feature, interaction_feature=None, 
                                 show=True, save_path=None):
    """
    Create a SHAP dependence plot for a single feature,
    optionally with an interaction feature
    
    Parameters:
    -----------
    model : estimator
        The trained model
    X : pandas DataFrame
        The input data
    feature : str
        The feature to plot
    interaction_feature : str, optional
        The interaction feature to include
    show : bool, default=True
        Whether to display the plot
    save_path : str, optional
        If provided, will save the plot to this path
    """
    import shap
    import matplotlib.pyplot as plt
    
    # Calculate SHAP values safely
    shap_values, expected_value = safe_shap_values(model, X)
    
    if shap_values is None:
        print("Could not calculate SHAP values")
        return
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Handle single feature or interaction
    if interaction_feature:
        shap.dependence_plot(feature, shap_values, X, interaction_index=interaction_feature, show=False)
    else:
        shap.dependence_plot(feature, shap_values, X, show=False)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def shap_force_plot(model, X, row_index=0, show=True, save_path=None):
    """
    Create a SHAP force plot for a single instance
    
    Parameters:
    -----------
    model : estimator
        The trained model
    X : pandas DataFrame
        The input data
    row_index : int or list, default=0
        The index of the row to explain, or a list of indices for multiple rows
    show : bool, default=True
        Whether to display the plot
    save_path : str, optional
        If provided, will save the plot to this path
    """
    import shap
    import matplotlib.pyplot as plt
    
    # Calculate SHAP values safely
    shap_values, expected_value = safe_shap_values(model, X)
    
    if shap_values is None:
        print("Could not calculate SHAP values")
        return
    
    # Convert single index to list if needed
    if isinstance(row_index, int):
        row_indices = [row_index]
    else:
        row_indices = row_index
    
    # Create the force plot
    force_plot = shap.force_plot(
        expected_value, 
        shap_values[row_indices], 
        X.iloc[row_indices]
    )
    
    # Save if requested
    if save_path:
        shap.save_html(save_path, force_plot)
    
    # Show if requested
    if show:
        return force_plot 
