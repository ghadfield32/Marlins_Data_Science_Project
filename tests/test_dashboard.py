#!/usr/bin/env python
"""
Test script to verify the explainer dashboard with our fixes.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.data.load_data import load_and_clean_data
from src.features.feature_engineering import feature_engineer
from src.utils.gbm_utils import load_pipeline, get_feature_importance
from src.utils.dash_utils import launch_explainer_dashboard
from src.data.ColumnSchema import _ColumnSchema

def main():
    print("Loading pipeline and data...")
    
    # 1) Try to load the model - check both paths since paths might differ
    try:
        model_path = "data/models/saved_models/gbm_pipeline.joblib"
        if not Path(model_path).exists():
            model_path = "models/gbm_pipeline.joblib"
        model_pipeline, preprocessor = load_pipeline(model_path)
        print(f"✅ Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 2) Load a data sample
    try:
        data_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
        df_raw = load_and_clean_data(data_path)
        
        # Use a small sample for testing
        df_raw = df_raw.sample(200, random_state=42)
        df_fe = feature_engineer(df_raw)
        X_raw = df_fe.drop(columns=["exit_velo"])
        y_raw = df_fe["exit_velo"]
        print(f"✅ Successfully loaded data: {len(df_raw)} rows, {len(df_fe.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 3) Get feature importance for debugging
    try:
        feature_imp = get_feature_importance(model_pipeline, preprocessor)
        print(f"✅ Got feature importance for {len(feature_imp)} features")
        print("Top 10 features:")
        print(feature_imp.head(10))
    except Exception as e:
        print(f"❌ Error getting feature importance: {e}")
    
    # 4) Initialize column schema
    cols = _ColumnSchema()
    
    # 5) Launch the dashboard
    print("\nLaunching explainer dashboard...\n")
    try:
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
            debug=True
        )
        print("\n✅ Dashboard launched successfully!")
    except Exception as e:
        print(f"\n❌ Error launching dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 