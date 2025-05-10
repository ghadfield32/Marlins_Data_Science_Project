#!/usr/bin/env python
"""
Standalone script to run the explainer dashboard with hostname resolution fix.
"""
import pandas as pd
from pathlib import Path

def main():
    print("\n=== Starting Dashboard with Hostname Resolution Fix ===\n")
    
    try:
        # Import the necessary modules
        from src.data.load_data import load_and_clean_data
        from src.features.feature_engineering import feature_engineer
        from src.utils.gbm_utils import load_pipeline
        from src.utils.dash_utils import launch_explainer_dashboard
        from src.data.ColumnSchema import _ColumnSchema
        
        # Try different possible model paths
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
        
        if not model_path:
            print("❌ Could not find the model file. Please specify the correct path.")
            return
            
        # Load the model and preprocessor
        print(f"Loading model from {model_path}...")
        model_pipeline, preprocessor = load_pipeline(model_path)
        print("✅ Model loaded successfully")
        
        # Load and prepare data
        data_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
        if not Path(data_path).exists():
            print(f"❌ Data file not found at {data_path}")
            return
            
        print(f"Loading data from {data_path}...")
        df_raw = load_and_clean_data(data_path)
        # Use a small sample for faster loading
        df_raw = df_raw.sample(200, random_state=42)
        df_fe = feature_engineer(df_raw)
        print(f"✅ Data loaded and processed: {len(df_raw)} rows")
        
        # Prepare X and y
        X_raw = df_fe.drop(columns=["exit_velo"])
        y_raw = df_fe["exit_velo"]
        
        # Get column schema
        cols = _ColumnSchema()
        
        # Launch the dashboard
        print("\n=== Launching Dashboard ===\n")
        print("If you encounter hostname resolution errors:")
        print("1. The dashboard is set to wait_for_ready=False by default to avoid these issues")
        print("2. If you still see errors, try setting a different host like 'localhost' or '0.0.0.0'\n")
        
        launch_explainer_dashboard(
            pipeline=model_pipeline,
            preprocessor=preprocessor,
            X_raw=X_raw,
            y=y_raw,
            cats=cols.nominal(),
            descriptions={c: c for c in preprocessor.get_feature_names_out()},
            whatif=True,
            shap_interaction=False,
            hide_wizard=True,
            debug=True,
            wait_for_ready=False,  # Disable health check to avoid hostname resolution errors
            title="Marlins Exit Velocity Model Explainer"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this script from the project root directory.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 