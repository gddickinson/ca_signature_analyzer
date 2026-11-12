"""Write tests/fixtures/cs_fixture.json: inputs plus the results of the repo's Python code on them.

The node test web/static/cs_engine.test.mjs runs the JS port (cs_*.js) on the same inputs and asserts agreement:
- derivatives() and stimulus_function() at sampled states/times (1e-12),
- ODE trajectories against odeint at tight tolerance (rtol 1e-12) for every stimulus and a custom-parameter run,
  plus the repo's own simulate() (default odeint tolerances) for the agreement figure in the README,
- extract_signature_features / estimate_bandwidth / estimate_snr / detect_oscillations on Python traces,
- entropy and mutual information on seeded integer and continuous data,
- the full web analysis (web.analysis.analyse_csv) on three CSV texts, and the signature-table analysis
  (web.signatures.analyse_table, JS-ported part) on the example CSV and the simulated demo table.

Run: python scripts/make_cs_fixture.py
"""
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.integrate import odeint

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web import analysis, signatures  # noqa: E402
from biophysical_model import CalciumDynamicsModel, ModelParameters  # noqa: E402
from information_theory import InformationAnalyzer  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "cs_fixture.json"
STIMULI = list(CalciumDynamicsModel.VALID_STIMULI)
CUSTOM = {"V_PM_pump": 60.0, "V_ER_release": 150.0, "k_IP3_deg": 0.3, "n_IP3": 2.5}


def derivative_cases(rng):
    out = []
    for k in range(40):
        stim = STIMULI[k % len(STIMULI)]
        params = CUSTOM if k % 3 == 0 else {}
        y = [float(rng.uniform(1, 5000)), float(rng.uniform(1e3, 1e5)), float(rng.uniform(1e4, 5e5)),
             float(rng.uniform(0, 500))]
        t = float(rng.uniform(0, 300))
        inten = float(rng.uniform(0, 2))
        m = CalciumDynamicsModel(ModelParameters(**params))
        out.append({"y": y, "t": t, "stimulus": stim, "intensity": inten, "params": params,
                    "dy": m.derivatives(np.array(y), t, stim, inten).tolist(),
                    "stim": m.stimulus_function(t, stim, inten)})
    return out


def ode_cases():
    out = []
    runs = [(s, 120.0, 0.5, 1.0, {}) for s in STIMULI] + [("H2O2", 200.0, 0.5, 1.5, CUSTOM)]
    for stim, dur, dt, inten, params in runs:
        m = CalciumDynamicsModel(ModelParameters(**params))
        t = np.arange(0, dur, dt)
        y0 = np.array([100.0, 50000.0, 300000.0, 10.0])
        sol = odeint(m.derivatives, y0, t, args=(stim, inten), rtol=1e-12, atol=1e-8, mxstep=100000)
        out.append({"stimulus": stim, "duration": dur, "dt": dt, "intensity": inten, "params": params,
                    "n": len(t), "y": sol.T.tolist()})
    return out


def repo_simulate_cases():
    """The repo's simulate() with its default odeint tolerances (600 s, dt 0.1), every 10th sample."""
    out = []
    m = CalciumDynamicsModel()
    for stim in STIMULI:
        t, sol = m.simulate(stim, duration=600.0, dt=0.1)
        out.append({"stimulus": stim, "stride": 10, "Ca_cyt": sol[::10, 0].tolist()})
    return out


def signal_cases():
    out = []
    m = CalciumDynamicsModel()
    for stim in STIMULI:
        t, sol = m.simulate(stim, duration=300.0, dt=0.1)
        ca = sol[:, 0]
        out.append({"name": stim, "dt": 0.1, "ca": ca.tolist(), **_signal(t, ca)})
    rng = np.random.default_rng(7)
    t = np.arange(0, 400, 0.25)
    ca = 120 + 400 * (np.sin(2 * np.pi * t / 37.0) > 0.7) + rng.normal(0, 15, t.size)
    out.append({"name": "noisy square oscillation (seeded)", "dt": 0.25, "ca": ca.tolist(), **_signal(t, ca)})
    t = np.arange(0, 100, 0.5)
    out.append({"name": "flat", "dt": 0.5, "ca": [100.0] * t.size, **_signal(t, np.full(t.size, 100.0))})
    return out


def _signal(t, ca):
    import utils
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        osc, freq = utils.detect_oscillations(t, ca)
        return analysis.clean({"features": CalciumDynamicsModel().extract_signature_features(t, ca),
                               "bandwidth": utils.estimate_bandwidth(t, ca), "snr": utils.estimate_snr(ca),
                               "oscillation": [osc, freq]})


def info_cases(rng):
    out = []
    for n, kx, ky in [(50, 3, 4), (300, 6, 2), (1000, 10, 10)]:
        x = rng.integers(0, kx, n)
        y = (x + rng.integers(0, ky, n)) % max(kx, ky)
        out.append({"kind": "discrete", "x": x.tolist(), "y": y.tolist(),
                    "hx": InformationAnalyzer.entropy_discrete(x),
                    "hx_mm": InformationAnalyzer.entropy_miller_madow(x),
                    "mi": InformationAnalyzer.mutual_information_discrete(x, y),
                    "mi_mm": InformationAnalyzer.mutual_information_discrete(x, y, bias_correction="miller_madow")})
    for n, nb in [(200, 10), (500, 7), (64, 3)]:
        x = rng.normal(size=n)
        y = x ** 2 + 0.3 * rng.normal(size=n)
        out.append({"kind": "continuous", "x": x.tolist(), "y": y.tolist(), "n_bins": nb,
                    "bins_x": analysis.binned(x, nb).tolist(),
                    "mi": InformationAnalyzer.mutual_information_continuous(x, y, n_bins=nb),
                    "mi_mm": InformationAnalyzer.mutual_information_continuous(x, y, n_bins=nb,
                                                                               bias_correction="miller_madow")})
    return out


def csv_cases(rng):
    m = CalciumDynamicsModel()
    t = np.arange(0, 300, 0.5)
    sims = {s: m.simulate(s, duration=300.0, dt=0.5)[1][:, 0] for s in ("ABA", "H2O2", "mechanical_touch")}
    lines = ["time_s," + ",".join(sims)] + [",".join([repr(float(t[i]))] + [repr(float(v[i])) for v in sims.values()])
                                            for i in range(t.size)]
    model_csv = "\n".join(lines) + "\n"
    t2 = np.arange(0, 240, 0.2)
    a = 0.1 + 0.6 * np.exp(-((t2 - 60) / 8.0) ** 2) + rng.normal(0, 0.01, t2.size)
    b = 0.1 + 0.3 * (np.sin(2 * np.pi * t2 / 25.0) > 0.8) + rng.normal(0, 0.02, t2.size)
    R = lambda v: repr(float(v))  # noqa: E731
    rows = [f"{R(t2[i])}\t{R(a[i])}\t{R(b[i])}" for i in range(t2.size)]
    rows[30] = f"{R(t2[30])}\t\t{R(b[30])}"  # missing value: row dropped
    tab_csv = "# seeded synthetic traces in uM\nt\tcell A\tcell B\n" + "\n".join(rows) + "\n"
    semi = "\n".join(f"{R(i)};{R(100 + 50 * np.sin(i / 3.0))}" for i in range(60))
    out = []
    for name, text, nb, units in [("model traces (simulated)", model_csv, 10, "nM"),
                                  ("tab-separated, uM, comment, missing value", tab_csv, 8, "uM"),
                                  ("headerless, semicolons", semi, 5, "nM")]:
        out.append({"name": name, "text": text, "n_bins": nb, "units": units,
                    "result": analysis.analyse_csv(text, nb, units)})
    return out


def table_cases():
    out = []
    ex = (ROOT / "data" / "example_signatures.csv").read_text()
    buf = io.StringIO()
    signatures.simulated_set().save_to_csv(buf)
    for name, text in [("example_signatures.csv", ex), ("simulated demo table", buf.getvalue())]:
        out.append({"name": name, "text": text,
                    "result": signatures.analyse_table(signatures.load_csv_text(text), full=False)})
    return out


def build():
    rng = np.random.default_rng(20261111)
    return {"derivatives": derivative_cases(rng), "ode": ode_cases(), "repo_simulate": repo_simulate_cases(),
            "signal": signal_cases(), "info": info_cases(rng), "csv": csv_cases(rng), "tables": table_cases()}


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), separators=(",", ":"), allow_nan=False))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")
