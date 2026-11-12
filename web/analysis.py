"""Trace analysis for the web interface, built only from the repo's own functions.

`parse_trace_csv(text)` reads a time column (s) plus one or more [Ca2+] columns; `analyse_traces(...)` returns, per
column, CalciumDynamicsModel.extract_signature_features, utils.estimate_bandwidth / estimate_snr /
detect_oscillations, the Shannon entropy of the uniformly binned trace (KBinsDiscretizer + entropy_discrete, as in
InformationAnalyzer.mutual_information_continuous), Shannon-Hartley capacity, and the steady-state (Hill) activation
of the default CDPK panel at baseline and peak; across columns, the binned mutual information matrix.
`simulate(...)` runs the biophysical model. The JS port (web/static/cs_*.js) mirrors this module and is tested
against it (tests/fixtures/cs_fixture.json, scripts/make_cs_fixture.py).
"""
import math
import warnings

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from web.model_spec import SRC  # noqa: F401  (puts src/ on sys.path)
from biophysical_model import CalciumDynamicsModel, ModelParameters  # noqa: E402
from decoder_model import CDPKDecoder, CDPKLibrary  # noqa: E402
from information_theory import InformationAnalyzer  # noqa: E402
import utils  # noqa: E402

MIN_ROWS = 20
UNIT_FACTOR = {"nM": 1.0, "uM": 1000.0}


def clean(v):
    """JSON-safe: NaN/inf -> None, numpy scalars -> Python."""
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return [clean(x) for x in v.tolist()]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else None
    return v


def _num(s):
    s = s.strip()
    if s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def parse_trace_csv(text):
    """Return (t, {name: values}, info). First column is time (s); later columns are [Ca2+] traces.

    Delimiter: comma, or tab / semicolon if the first line has those and no comma. Blank lines and lines starting
    with '#' are skipped. A first line that is not all numbers is the header. Rows with any empty or non-numeric
    cell are dropped (counted in info['dropped']).
    """
    lines = [ln for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
             if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError("The file is empty.")
    first = lines[0]
    delim = "," if "," in first else ("\t" if "\t" in first else (";" if ";" in first else ","))
    head = [c.strip().strip('"') for c in first.split(delim)]
    if all(_num(c) is not None for c in head):
        names = ["time"] + [f"trace_{i}" for i in range(1, len(head))]
    else:
        names, lines = head, lines[1:]
    if len(names) < 2:
        raise ValueError("Need at least two columns: time (s) and one [Ca2+] trace.")
    rows, dropped = [], 0
    for ln in lines:
        cells = [_num(c.strip().strip('"')) for c in ln.split(delim)]
        if len(cells) < len(names) or any(c is None for c in cells[:len(names)]):
            dropped += 1
            continue
        rows.append(cells[:len(names)])
    if len(rows) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} complete rows; found {len(rows)}.")
    arr = np.array(rows, dtype=float)
    t = arr[:, 0]
    dts = np.diff(t)
    if np.any(dts <= 0):
        raise ValueError("The time column (first column) must increase strictly.")
    med = float(np.median(dts))
    uniform = bool(np.max(np.abs(dts - med)) <= 1e-3 * med)
    cols = {names[j] or f"trace_{j}": arr[:, j] for j in range(1, len(names))}
    return t, cols, {"rows": len(rows), "dropped": dropped, "dt": float(t[1] - t[0]), "uniform": uniform,
                     "time_column": names[0]}


def binned(x, n_bins):
    """KBinsDiscretizer(uniform, ordinal) codes, as InformationAnalyzer.mutual_information_continuous bins."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
        return disc.fit_transform(np.asarray(x, float).reshape(-1, 1)).astype(int).flatten()


def analyse_column(t, ca, n_bins=10):
    model = CalciumDynamicsModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = model.extract_signature_features(t, ca)
        bandwidth = utils.estimate_bandwidth(t, ca)
        snr = utils.estimate_snr(ca)
        osc, freq = utils.detect_oscillations(t, ca)
    snr_power = snr * snr
    capacity = InformationAnalyzer.channel_capacity_shannon(bandwidth, snr_power)
    entropy = InformationAnalyzer.entropy_discrete(binned(ca, n_bins))
    cdpk = []
    for p in CDPKLibrary.get_default_cdpks():
        d = CDPKDecoder(p)
        cdpk.append({"name": p.name, "K_d": p.K_d, "n_Hill": p.n_Hill,
                     "at_baseline": d.steady_state_activation(feats["baseline_ca"]),
                     "at_peak": d.steady_state_activation(feats["peak_amplitude"])})
    return {"features": feats, "bandwidth_hz": bandwidth, "snr_amplitude": snr, "snr_power": snr_power,
            "capacity_bits_per_s": capacity, "oscillation": {"oscillatory": osc, "frequency": freq},
            "entropy_bits": entropy, "max_entropy_bits": math.log2(n_bins), "cdpk": cdpk}


def analyse_traces(t, cols, n_bins=10, units="nM"):
    if units not in UNIT_FACTOR:
        raise ValueError(f"units must be one of {list(UNIT_FACTOR)}")
    if not 2 <= int(n_bins) <= 100:
        raise ValueError("n_bins must be between 2 and 100.")
    n_bins = int(n_bins)
    f = UNIT_FACTOR[units]
    t = np.asarray(t, float)
    data = {k: np.asarray(v, float) * f for k, v in cols.items()}
    names = list(data)
    out = [{"name": k, **analyse_column(t, data[k], n_bins)} for k in names]
    mi = [[0.0] * len(names) for _ in names]
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j >= i:
                v = InformationAnalyzer.mutual_information_continuous(data[a], data[b], n_bins=n_bins)
                mi[i][j] = mi[j][i] = v
    return clean({"n_bins": n_bins, "units": units, "columns": out, "mi": {"names": names, "matrix": mi}})


def analyse_csv(text, n_bins=10, units="nM"):
    t, cols, info = parse_trace_csv(text)
    res = analyse_traces(t, cols, n_bins, units)
    res["info"] = info
    res["t"] = t.tolist()
    res["traces"] = {k: (v * UNIT_FACTOR[units]).tolist() for k, v in cols.items()}
    if not info["uniform"]:
        res["warnings"] = ["Time steps are not uniform: features use t[1] - t[0] as the sampling interval."]
    return res


def simulate(stimulus="ABA", duration=600.0, dt=0.1, intensity=1.0, params=None):
    """Run CalciumDynamicsModel.simulate; return the four state traces and the signature features."""
    if not 0 < float(duration) <= 3600:
        raise ValueError("duration must be in (0, 3600] s.")
    if not 0.01 <= float(dt) <= 10:
        raise ValueError("dt must be in [0.01, 10] s.")
    known = {f for f in ModelParameters.__dataclass_fields__}
    bad = set(params or {}) - known
    if bad:
        raise ValueError(f"Unknown parameters: {sorted(bad)}")
    model = CalciumDynamicsModel(ModelParameters(**{k: float(v) for k, v in (params or {}).items()}))
    t, sol = model.simulate(stimulus, duration=float(duration), dt=float(dt), intensity=float(intensity))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = model.extract_signature_features(t, sol[:, 0])
    return clean({"stimulus": stimulus, "t": t, "Ca_cyt": sol[:, 0], "Ca_ER": sol[:, 1], "Ca_vac": sol[:, 2],
                  "IP3": sol[:, 3], "features": feats})
