"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests ----------------------------------------------------------------------------------
import json  # noqa: E402

import numpy as np  # noqa: E402

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "cs_fixture.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text())


def _close(a, b, rtol, path="$", atol=1e-9):
    """Recursive comparison with a tolerance (CI runs on Linux, whose libm differs in the last digit)."""
    if isinstance(b, dict):
        assert set(a) == set(b), path
        for k in b:
            _close(a[k], b[k], rtol, f"{path}.{k}", atol)
    elif isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, rtol, f"{path}[{i}]", atol)
    elif isinstance(b, float) and not isinstance(b, bool):
        assert np.isclose(a, b, rtol=rtol, atol=atol), f"{path}: {a} vs {b}"
    elif isinstance(b, str) and a != b and ("," in b or "\t" in b):
        # generated CSV/TSV text: numbers formatted from floats can differ in the last digit on Linux
        ra, rb = a.strip().splitlines(), b.strip().splitlines()
        assert len(ra) == len(rb), path
        for i, (la, lb) in enumerate(zip(ra, rb)):
            ca, cb = re.split(r"[,\t]", la), re.split(r"[,\t]", lb)
            assert len(ca) == len(cb), f"{path} line {i}"
            for x, y in zip(ca, cb):
                try:
                    fx, fy = float(x), float(y)
                except ValueError:
                    assert x == y, f"{path} line {i}"
                    continue
                assert np.isclose(fx, fy, rtol=rtol, atol=atol), f"{path} line {i}: {x} vs {y}"
    else:
        assert a == b, path


def test_fixture_is_current():
    """scripts/make_cs_fixture.py would write the same numbers (so the JS test checks today's Python)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_cs_fixture
    fresh = json.loads(json.dumps(make_cs_fixture.build()))
    for key, value in FIXTURE.items():
        # ODE trajectories: odeint's own absolute tolerance is ~1e-8, so values near zero
        # differ by ~1e-9 between macOS and the Linux CI runner and may change sign; compare them with atol 1e-8.
        atol = 1e-8 if key in ("ode", "repo_simulate") else 1e-9
        _close(fresh[key], value, 1e-6, key, atol)


@pytest.mark.parametrize("case", FIXTURE["csv"], ids=lambda c: c["name"])
def test_analyze_matches_fixture(case):
    r = client.post("/api/analyze", json={"text": case["text"], "n_bins": case["n_bins"], "units": case["units"]})
    assert r.status_code == 200, r.text
    _close(r.json(), case["result"], 1e-9)


def test_analyze_rejects_bad_input():
    assert client.post("/api/analyze", json={"text": "t,a\n0,1\n1,2\n"}).status_code == 400
    bad_time = "t,a\n" + "".join(f"{(i * 7) % 20},{i}\n" for i in range(30))
    r = client.post("/api/analyze", json={"text": bad_time})
    assert r.status_code == 400 and "increase" in r.text


def test_simulate_matches_model():
    r = client.post("/api/simulate", json={"stimulus": "NaCl", "duration": 60, "dt": 0.5})
    assert r.status_code == 200
    d = r.json()
    assert len(d["t"]) == 120 and d["Ca_cyt"][0] == 100.0
    assert set(d["features"]) >= {"baseline_ca", "peak_amplitude", "n_peaks"}
    assert client.post("/api/simulate", json={"stimulus": "light"}).status_code == 400
    assert client.post("/api/simulate", json={"stimulus": "ABA", "params": {"nope": 1}}).status_code == 400


def test_signature_endpoints():
    db = client.get("/api/database").json()
    assert db["summary"]["total_signatures"] == 7 and len(db["discrimination"]) == 7
    demo = client.get("/api/demo_signatures").json()
    assert demo["summary"]["total_signatures"] == 35
    assert 0 <= demo["python"]["classification"]["accuracy_mean"] <= 1
    text = (ROOT / "data" / "example_signatures.csv").read_text()
    up = client.post("/api/signatures", json={"text": text}).json()
    assert up["features"] == db["features"]
    assert client.post("/api/signatures", json={"text": "a,b\n1,2\n"}).status_code == 400
