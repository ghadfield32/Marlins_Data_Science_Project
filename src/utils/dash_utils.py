
import dash
from src.utils.dash_compat import patch_dropdown_right_once
import pandas as pd
import dash_bootstrap_components as dbc
from explainerdashboard import RegressionExplainer, ExplainerDashboard
from __future__ import annotations
import socket, contextlib, time

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

def launch_explainer_dashboard(
    pipeline,
    preprocessor,
    X_raw: pd.DataFrame,
    y: pd.Series,
    *,
    cats=None,
    descriptions=None,
    title: str = "Model Explainer",
    bootstrap=dbc.themes.FLATLY,
    inline: bool | None = None,
    host: str = "127.0.0.1",
    port: int = 8050,
    **db_kwargs,
):
    """Create & display an ExplainerDashboard.

    • Inline in notebooks by default.  
    • Auto-terminates any previous inline dashboard on the same port.  
    • Falls back to a free port if the requested one is still busy.
    """
    if inline is None:
        inline = _in_notebook()

    # ---------- tidy up any existing inline server ----------
    if inline and _port_in_use(host, port):
        try:
            ExplainerDashboard.terminate(port)
        except Exception:             # pragma: no cover
            pass                      # nothing running or unsupported version
        # give the OS a moment to release the socket
        for _ in range(10):
            if not _port_in_use(host, port):
                break
            time.sleep(0.1)
        else:                          # still busy after 1 s
            port = _get_free_port()

    # ---------- prepare dataset ----------
    X_proc     = preprocessor.transform(X_raw)
    feat_names = preprocessor.get_feature_names_out()
    X_df       = pd.DataFrame(X_proc, columns=feat_names, index=X_raw.index)

    if cats:
        cats = [c for c in cats if any(fn.startswith(f"{c}_") for fn in feat_names)]

    explainer = RegressionExplainer(
        pipeline,
        X_df,
        y,
        cats=cats or [],
        descriptions=descriptions or {},
        precision="float32",
    )
    explainer.merged_cols = explainer.merged_cols.intersection(X_df.columns)

    # ---------- launch dashboard ----------
    if inline:
        ExplainerDashboard(
            explainer,
            title=title,
            bootstrap=bootstrap,
            mode="inline",
            **db_kwargs,
        ).run(host=host, port=port)
    else:
        ExplainerDashboard(
            explainer,
            title=title,
            bootstrap=bootstrap,
            **db_kwargs,
        ).run(host=host, port=port)







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
        hide_wizard     = True
    )
