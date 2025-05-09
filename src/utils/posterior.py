# src/utils/posterior.py
import numpy as np
import pandas as pd
import arviz as az

# ── REPLACEMENT: paste over the whole function ─────────────────────────
import json, pathlib, numpy as np, pandas as pd, arviz as az

JSON_GLOBAL = pathlib.Path("data/models/saved_models/global_effects.json")

def posterior_to_frame(idata: az.InferenceData) -> pd.DataFrame:
    """
    Returns a batter‑level summary **AND** writes global effects to JSON.

    File written  ➜  data/models/saved_models/global_effects.json
    """
    post = idata.posterior

    # ------- per‑batter u summaries -----------------------------------
    u   = post["u"]                                         # (chain,draw,batter)
    stats = {
        "u_mean"  : u.mean(("chain","draw")).values,
        "u_sd"    : u.std (("chain","draw")).values,
        "u_q2.5"  : np.percentile(u.values,  2.5, axis=(0,1)),
        "u_q50"   : np.percentile(u.values, 50.0, axis=(0,1)),
        "u_q97.5" : np.percentile(u.values,97.5, axis=(0,1)),
    }
    df = pd.DataFrame({"batter_idx": np.arange(u.shape[-1]), **stats})

    # ------- global effects ------------------------------------------
    mu_mean = post["mu"].mean().item()

    # β_age  ➜ last entry of beta vector (age_centered was added last)
    beta  = post["beta"]
    feat_dim = [d for d in beta.dims if d not in ("chain","draw")][0]
    beta_age = beta.isel({feat_dim: -1}).mean().item()

    beta_level = post["beta_level"].mean(("chain","draw")).values.tolist()
    sigma_b    = post["sigma_b"].mean().item()
    sigma_e    = post["sigma_e"].mean().item()

    global_eff = dict(
        mu_mean=mu_mean,
        beta_age=beta_age,
        beta_level=beta_level,
        sigma_b=sigma_b,
        sigma_e=sigma_e,
        median_age=idata.attrs.get("median_age", 26.0),
        beta=post["beta"].mean(("chain","draw")).values.tolist(),  # Save all beta coefficients
        feature_names=idata.attrs.get("feature_names", [])  # Also save feature names if available
    )

    # ➜  write side‑car JSON (overwrite every run)
    JSON_GLOBAL.write_text(json.dumps(global_eff, indent=2))
    print(f"✔︎ wrote global effects → {JSON_GLOBAL}")

    return df





def align_batter_codes(df_roster: pd.DataFrame,
                       train_cats: pd.Index) -> pd.Series:
    """
    Convert integer batter_ids in *roster* into the categorical codes
    **identical** to what the model saw during training.

    Any unseen batter gets code = -1 (handled later).
    """
    cat = pd.Categorical(df_roster["batter_id"], categories=train_cats)
    return pd.Series(cat.codes, index=df_roster.index)

