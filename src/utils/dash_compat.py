# src/utils/dash_compat.py
import dash
import functools
import dash_bootstrap_components as dbc

# ── 3.0 compat: if Dash.run_server() was removed, alias it to run()
if not hasattr(dash.Dash, "run_server"):
    dash.Dash.run_server = dash.Dash.run

_DROPDOWN_PATCH_KEY = "_ghadf_dropdown_patched"

def patch_dropdown_right_once() -> None:
    """
    Back-compat shim: translate deprecated `right=` → `align_end=`,
    and patch it exactly once per Python process.
    """
    if getattr(dbc, _DROPDOWN_PATCH_KEY, False):
        return

    orig_init = dbc.DropdownMenu.__init__

    @functools.wraps(orig_init)
    def _init(self, *args, **kwargs):
        if "right" in kwargs and "align_end" not in kwargs:
            kwargs["align_end"] = kwargs.pop("right")
        return orig_init(self, *args, **kwargs)

    dbc.DropdownMenu.__init__ = _init
    setattr(dbc, _DROPDOWN_PATCH_KEY, True)

