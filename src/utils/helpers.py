import pandas as pd
import numpy as np


def compute_league_prior(df):
    """
    Compute the league-wide prior mean and standard deviation.
    
    Args:
        df: DataFrame with exit velocity data
        
    Returns:
        Tuple of (prior_mean, prior_sd)
    """
    # Simple implementation: overall mean and SD
    prior_mean = df["exit_velo"].mean()
    prior_sd = df["exit_velo"].std()
    
    return prior_mean, prior_sd


def estimate_noise_variance(df, group_col="batter_id"):
    """
    Estimate within-batter variance of exit velocity.
    
    Args:
        df: DataFrame with exit velocity data
        group_col: Column to group by (default: batter_id)
        
    Returns:
        Estimated noise variance
    """
    # Group by batter and calculate variance within each group
    group_vars = df.groupby(group_col)["exit_velo"].var()
    
    # Take the mean of these variances as an estimate of noise variance
    noise_var = group_vars.mean()
    
    return noise_var 