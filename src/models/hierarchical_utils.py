
import arviz as az

def save_model(idata, file_path: str, overwrite: bool = True):
    """Save ArviZ InferenceData to NetCDF."""
    idata.to_netcdf(file_path, engine="h5netcdf", overwrite_existing=overwrite)
    print(f"✔︎ saved model → {file_path}")

def load_model(file_path: str):
    """Load ArviZ InferenceData from NetCDF."""
    idata = az.from_netcdf(file_path, engine="h5netcdf")
    print(f"✔︎ loaded model ← {file_path}")
    return idata

if __name__ == "__main__":
    # === Editable settings ===
    # Path to the saved model (NetCDF format)
    MODEL_PATH = "data/models/saved_models/model.nc"
    # Input data for prediction (raw CSV with exit velocity data)
    raw_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
    # Output predictions file (CSV) or set to None to print to console
    OUTPUT_PREDS_2024 = "data/predictions/predictions_2024.csv"  # <-- EDITABLE: set output CSV path or None
    
    model = load_model(MODEL_PATH)
    save_model(model, MODEL_PATH)
