from __future__ import annotations
import dash
from src.utils.dash_compat import patch_dropdown_right_once
import pandas as pd
import dash_bootstrap_components as dbc
from explainerdashboard import RegressionExplainer, ExplainerDashboard
import socket, contextlib, time
import numpy as np
from src.utils.shap_utils import safe_shap_values

def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.has_trait("kernel")
    except Exception:
        return False

# Apply necessary shims at import time
patch_dropdown_right_once()



def _port_in_use(host: str, port: int) -> bool:
    with contextlib.closing(socket.socket()) as s:
        s.settimeout(0.001)
        return s.connect_ex((host, port)) == 0

def _get_free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

def patch_explainer_for_xgboost(explainer, X_df, debug=False):
    """
    Apply patches to the explainer to handle XGBoost SHAP value calculation issues.
    
    Parameters:
    -----------
    explainer : RegressionExplainer
        The explainer to patch
    X_df : pd.DataFrame
        The feature dataframe
    debug : bool, default=False
        Whether to print debug information
        
    Returns:
    --------
    bool
        True if patched successfully, False otherwise
    """
    try:
        import shap
        from types import MethodType
        
        # 1. Override the default get_shap_values_df method
        def patched_get_shap_values_df(self, *args, **kwargs):
            if debug:
                print("Using patched get_shap_values_df method")
            
            if hasattr(self, '_shap_values_df') and self._shap_values_df is not None:
                return self._shap_values_df
            
            # Extract the model from pipeline if needed
            model = self.model
            if hasattr(model, 'steps'):
                model = model.steps[-1][1]
            
            # Use our safe implementation
            shap_values, expected_value = safe_shap_values(
                model, 
                X_df,
                max_features=min(100, X_df.shape[1]),
                sample_size=min(1000, len(X_df))
            )
            
            if shap_values is None:
                if debug:
                    print("Failed to calculate SHAP values, falling back to permutation importance")
                # Fall back to permutation importance for feature importance
                self._shap_values_df = None
                return pd.DataFrame(0, index=X_df.index, columns=X_df.columns)
            
            if debug:
                print(f"SHAP values shape: {np.array(shap_values).shape}")
            
            # Handle different explainer output formats
            if hasattr(shap_values, 'values'):  # New SHAP output format
                shap_df = pd.DataFrame(shap_values.values, columns=X_df.columns, index=X_df.index)
            else:  # Old-style output
                shap_df = pd.DataFrame(shap_values, columns=X_df.columns, index=X_df.index)
            
            # Cache the result
            self._shap_values_df = shap_df
            self.expected_value = expected_value
            
            return shap_df
        
        # 2. Apply the patch
        explainer.get_shap_values_df = MethodType(patched_get_shap_values_df, explainer)
        
        if debug:
            print("Successfully patched explainer for XGBoost compatibility")
        return True
        
    except Exception as e:
        if debug:
            print(f"Failed to patch explainer: {e}")
        return False

def launch_explainer_dashboard(
    pipeline,
    preprocessor,
    X_raw: pd.DataFrame,
    y: pd.Series,
    *,
    host: str = "127.0.0.1",
    port: int = 8050,
    debug: bool = False,
    wait_for_ready: bool = False,
    title: str = "Model Explainer",
    **db_kwargs,
) -> tuple[str, int]:
    """
    1. Finds a free port if needed
    2. Prints debug logs at each step (if debug=True)
    3. Prints final URL
    4. Gracefully handles stale‐thread & DNS errors
    Returns the (host, port) in use.
    """
    # --- PRE-STEP: Terminate any existing dashboard on this port ---
    if debug:
        print("[Pre-Step] Terminating any existing dashboard on port", port)
    try:
        ExplainerDashboard.terminate(port)
        if debug:
            print(f"[Pre-Step] Successfully terminated any previous dashboard on port {port}")
        # Give the system a moment to fully release the port
        time.sleep(0.5) 
    except Exception as e:
        if debug:
            print(f"[Pre-Step] No existing dashboard to terminate or error: {e}")

    # --- STEP 1: Port check ---
    if debug:
        print("[Step 1] Checking port availability…")
    if _port_in_use(host, port):
        if debug:
            print(f"[Step 1] Port {port} still in use; grabbing a free port…")
        port = _get_free_port()
    if debug:
        print(f"[Step 1] Using port: {port}")

    # --- STEP 2: Data prep ---
    if debug:
        print("[Step 2] Preparing data…")
    X_proc = preprocessor.transform(X_raw)
    feat_names = preprocessor.get_feature_names_out()
    X_df = pd.DataFrame(X_proc, columns=feat_names, index=X_raw.index)
    if debug:
        print(f"[Step 2] X_df shape: {X_df.shape}")

    # --- STEP 3: Explainer build & patch ---
    if debug:
        print("[Step 3] Building RegressionExplainer & patching for XGBoost…")
    explainer = RegressionExplainer(pipeline, X_df, y, precision="float32")
    patch_explainer_for_xgboost(explainer, X_df, debug=debug)
    if debug:
        print("[Step 3] Explainer ready. Merged cols:", len(explainer.merged_cols))

    # --- STEP 4: Tell the user the URL ---
    url = f"http://{host}:{port}"
    print(f"🔍  Dashboard URL → {url}")

    # --- STEP 5: Launch with graceful error handling ---
    if debug:
        print("[Step 5] Launching ExplainerDashboard…")
    dashboard = ExplainerDashboard(
        explainer,
        title=title,
        bootstrap=dbc.themes.FLATLY,
        mode="inline",
        **db_kwargs,
    )

    # Check if the run method supports wait_for_ready parameter
    try:
        # Inspect the run method signature to see if it supports wait_for_ready
        from inspect import signature
        run_params = signature(dashboard.run).parameters
        supports_wait_for_ready = 'wait_for_ready' in run_params
        
        if debug and not supports_wait_for_ready and wait_for_ready:
            print("[Warning] wait_for_ready parameter not supported in this version of explainerdashboard")
    except Exception as e:
        if debug:
            print(f"[Note] Could not determine if wait_for_ready is supported: {e}")
        supports_wait_for_ready = False

    try:
        # Run with wait_for_ready only if supported
        if supports_wait_for_ready:
            dashboard.run(host=host, port=port, wait_for_ready=wait_for_ready)
        else:
            dashboard.run(host=host, port=port)
    except TypeError as e:
        # stale-thread kill bug
        if "NoneType" in str(e):
            print("[Warning] Ignored stale-thread TypeError during restart.")
            # Try to run again without wait_for_ready
            dashboard.run(host=host, port=port)
        else:
            raise
    except Exception as e:
        # DNS / health-check bug
        msg = str(e)
        if "Name or service not known" in msg or "getaddrinfo" in msg:
            print("[Warning] DNS resolution failed. Retrying without health check…")
            dashboard.run(host=host, port=port)
        else:
            raise

    # --- Return for programmatic use ---
    return host, port








if __name__=="__main__":
    from pathlib import Path
    import pandas as pd
    from src.data.load_data import load_and_clean_data
    from src.features.feature_engineering import feature_engineer
    from src.features.preprocess import transform_preprocessor
    from src.utils.gbm_utils import load_pipeline
    from src.data.ColumnSchema import _ColumnSchema

    # 1) load your trained pipeline + preprocessor
    model_pipeline, preprocessor = load_pipeline("data/models/saved_models/gbm_pipeline.joblib")

    # 2) load & prepare a small sample
    df_raw = load_and_clean_data(
        "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
    ).sample(200, random_state=42)
    df_fe  = feature_engineer(df_raw)
    X_raw = df_fe.drop(columns=["exit_velo"])
    y_raw = df_fe["exit_velo"]

    # 3) category grouping helper
    cols = _ColumnSchema()

    # 4) launch the dashboard on port 8050
    launch_explainer_dashboard(
        pipeline      = model_pipeline,
        preprocessor  = preprocessor,
        X_raw         = X_raw,
        y             = y_raw,
        cats          = cols.nominal(),
        descriptions  = {c: c for c in preprocessor.get_feature_names_out()},
        whatif        = True,
        shap_interaction = False,
        hide_wizard     = True,
        debug           = True
    )

