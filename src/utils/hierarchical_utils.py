
import arviz as az
import joblib
from pathlib import Path

def save_model(idata, file_path: str, overwrite: bool = True):
    """Save ArviZ InferenceData to NetCDF."""
    idata.to_netcdf(file_path, engine="h5netcdf", overwrite_existing=overwrite)
    print(f"✔︎ saved model → {file_path}")

def load_model(file_path: str):
    """Load ArviZ InferenceData from NetCDF."""
    idata = az.from_netcdf(file_path, engine="h5netcdf")
    print(f"✔︎ loaded model ← {file_path}")
    return idata

def save_preprocessor(transformer, file_path: str):
    """Save the scikit-learn transformer to a joblib file."""
    joblib.dump(transformer, file_path)
    print(f"✔︎ saved preprocessor → {file_path}")

def load_preprocessor(file_path: str):
    """Load the scikit-learn transformer from a joblib file."""
    transformer = joblib.load(file_path)
    print(f"✔︎ loaded preprocessor ← {file_path}")
    return transformer

if __name__ == "__main__":
    # === Editable settings ===
    # Path to the saved model (NetCDF format)
    MODEL_PATH = "data/models/saved_models/model.nc"
    # Path to the preprocessor
    PREPROC_PATH = "data/models/saved_models/preprocessor.joblib"
    # Input data for prediction (raw CSV with exit velocity data)
    raw_path = "data/Research Data Project/Research Data Project/exit_velo_project_data.csv"
    # Output predictions file (CSV) or set to None to print to console
    OUTPUT_PREDS_2024 = "data/predictions/predictions_2024.csv"  # <-- EDITABLE: set output CSV path or None

    model = load_model(MODEL_PATH)
    save_model(model, MODEL_PATH)

