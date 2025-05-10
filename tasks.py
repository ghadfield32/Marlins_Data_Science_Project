# invoke.py: Task definitions using Invoke (install with `pip install invoke`)
from invoke import task

@task
def gpu_test(c):
    """Run a basic JAX GPU test (print device info)."""
    c.run("python3 - << 'EOF'\n"
          "import jax; print('JAX version:', jax.__version__);\n"
          "print('Devices:', jax.devices());\n"
          "print('GPU count:', jax.device_count('gpu'))\n"
          "EOF")

@task
def hierarchical(c, draws=1000, tune=1000):
    """Fit the hierarchical Bayesian model (PyMC) with JAX/GPU."""
    cmd = f"python3 gpu_testing/run_hierarchical_direct.py --draws {draws} --tune {tune}"
    c.run(cmd)

@task
def jupyter(c, port=8888):
    """Launch JupyterLab on given port."""
    c.run(f"jupyter lab --ip=0.0.0.0 --port={port} --no-browser --allow-root")

@task
def explainerdash(c, host="127.0.0.1", port=8050, debug=False):
    """Launch the model explainer dashboard on the specified port."""
    debug_flag = "--debug" if debug else ""
    c.run(f"python src/run_explainer_dash.py --host {host} --port {port} {debug_flag}")

@task
def convert_to_parquet(c, csv_path="data/exit_velo_project_data.csv"):
    """Convert a CSV file to Parquet format for faster loading."""
    c.run(f"python3 -c \"import pandas as pd; \
            df = pd.read_csv('{csv_path}'); \
            parquet_path = '{csv_path}'.replace('.csv', '.parquet'); \
            df.to_parquet(parquet_path, index=False); \
            print(f'Converted {csv_path} to {parquet_path}')\"")

@task
def lint(c):
    """Run code linting."""
    c.run("flake8 src/")

@task
def test(c):
    """Run tests."""
    c.run("pytest")

@task
def clean(c):
    """Clean up temporary files."""
    c.run("find . -type d -name __pycache__ -exec rm -rf {} +")
    c.run("find . -type f -name '*.pyc' -delete")
    c.run("find . -type f -name '*.pyo' -delete")
    c.run("find . -type f -name '*.pyd' -delete")
    c.run("find . -type f -name '.coverage' -delete")
    c.run("rm -rf .pytest_cache")
    c.run("rm -rf .coverage")
    c.run("rm -rf htmlcov")
    c.run("rm -rf dist")
    c.run("rm -rf build")

@task
def rebuild(c):
    """Re-build the Docker image & recreate dev container."""
    c.run("docker compose build --no-cache") 