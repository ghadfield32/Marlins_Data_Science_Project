#!/usr/bin/env python
"""
Test functionality of the explainer dashboard.
Tests:
1. Dashboard can be launched
2. Dashboard can be relaunched on the same port
3. Port detection and cleanup works correctly
"""
import time
import pytest
import pandas as pd
from pathlib import Path


def test_dashboard_launch_and_relaunch():
    """Test that the dashboard can be launched and relaunched on the same port."""
    try:
        # Import the necessary modules
        from src.data.load_data import load_and_clean_data
        from src.features.feature_engineering import feature_engineer
        from src.utils.gbm_utils import load_pipeline
        from src.utils.dash_utils import launch_explainer_dashboard, _port_in_use
        from src.data.ColumnSchema import _ColumnSchema
        
        # Find the model
        model_paths = [
            "data/models/saved_models/gbm_pipeline.joblib",
            "models/gbm_pipeline.joblib",
            "data/models/gbm_pipeline.joblib"
        ]
        
        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                break
        
        assert model_path is not None, "Could not find model file"
            
        # Load the model and preprocessor
        model_pipeline, preprocessor = load_pipeline(model_path)
        
        # Load and prepare data
        data_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
        assert Path(data_path).exists(), f"Data file not found at {data_path}"
            
        df_raw = load_and_clean_data(data_path)
        # Use a very small sample for quick testing
        df_raw = df_raw.sample(20, random_state=42)
        df_fe = feature_engineer(df_raw)
        
        # Prepare X and y
        X_raw = df_fe.drop(columns=["exit_velo"])
        y_raw = df_fe["exit_velo"]
        
        # Get column schema
        cols = _ColumnSchema()
        
        # Specify a test port 
        test_port = 8099  # Use a high port number unlikely to be in use
        
        # === First Launch ===
        host1, port1 = launch_explainer_dashboard(
            pipeline=model_pipeline,
            preprocessor=preprocessor,
            X_raw=X_raw,
            y=y_raw,
            port=test_port,
            cats=cols.nominal(),
            descriptions={c: c for c in preprocessor.get_feature_names_out()},
            whatif=True,
            shap_interaction=False,
            hide_wizard=True,
            debug=True,
            wait_for_ready=False,
            title="Test Dashboard (First Launch)"
        )
        
        # Verify port is in use
        assert _port_in_use("127.0.0.1", port1), f"Port {port1} should be in use after first launch"
        
        # === Second Launch (should close first then relaunch) ===
        host2, port2 = launch_explainer_dashboard(
            pipeline=model_pipeline,
            preprocessor=preprocessor,
            X_raw=X_raw,
            y=y_raw,
            port=test_port,  # Same port as before
            cats=cols.nominal(),
            descriptions={c: c for c in preprocessor.get_feature_names_out()},
            whatif=True,
            shap_interaction=False,
            hide_wizard=True,
            debug=True,
            wait_for_ready=False,
            title="Test Dashboard (Second Launch)"
        )
        
        # Verify port is still in use
        assert _port_in_use("127.0.0.1", port2), f"Port {port2} should be in use after second launch"
        
        # Verify we got the same port both times
        assert port1 == port2, f"Expected same port both times, got: {port1} and {port2}"
        assert port1 == test_port, f"Expected port {test_port}, got: {port1}"
        
        # Clean up - terminate the dashboard
        from explainerdashboard import ExplainerDashboard
        ExplainerDashboard.terminate(port2)
        time.sleep(1)  # Give it time to shut down
        
        # Verify port is no longer in use
        assert not _port_in_use("127.0.0.1", port2), f"Port {port2} should be free after termination"
        
    except Exception as e:
        pytest.fail(f"Error during test: {e}")

if __name__ == "__main__":
    # Run the test directly when script is executed
    test_dashboard_launch_and_relaunch() 