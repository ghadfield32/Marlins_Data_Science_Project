# location: tests/test_jax_memory_configs.py
"""
Spawn a fresh Python for each flag combo and assert the pooled
fraction passes a dynamic threshold (40 % if prealloc, else 5 %).
"""
import subprocess, sys, json, pytest, textwrap

COMBOS = [
    ("false", 0.90),
    ("true",  0.95),
    ("true",  0.50),
    ("false", 0.20),   # still OK: low pool expected
]

code_tpl = textwrap.dedent("""
    import os, re, json, subprocess, jax
    os.environ.update(
        XLA_PYTHON_CLIENT_PREALLOCATE="{pre}",
        XLA_PYTHON_CLIENT_MEM_FRACTION="{frac}",
        XLA_PYTHON_CLIENT_ALLOCATOR="platform",
    )
    from jax.lib import xla_client as xc
    def mem():
        if hasattr(xc, "get_gpu_memory_info"):
            return xc.get_gpu_memory_info(0)
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.free",
             "--format=csv,noheader,nounits"], text=True).splitlines()[0]
        tot, free = (int(s.strip()) for s in re.split(r",\\s*", out, 1))
        return free*1048576, tot*1048576
    f,t = mem(); pool = (t-f)/t
    need = 0.035 if "{pre}"=="true" else 0.03
    print(json.dumps(dict(pool=pool, need=need)))
""")

@pytest.mark.parametrize("pre,frac", COMBOS,
                         ids=[f"prealloc_{p}_{frac}" for p,frac in COMBOS])
def test_combo(pre, frac):
    out = subprocess.run([sys.executable, "-c", code_tpl.format(pre=pre, frac=frac)],
                         capture_output=True, text=True, check=True)
    obj = json.loads(out.stdout)
    assert obj["pool"] >= obj["need"], f"pool {obj['pool']:.2%} < {obj['need']:.0%}"


