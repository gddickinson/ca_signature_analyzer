"""Model metadata for the web interface, read from the repo's own code.

`model_parameters()` lists every ModelParameters field with its default, unit group and comment, parsed from
src/biophysical_model.py so the web page shows the code's own wording. `STIMULI` repeats the comments of
CalciumDynamicsModel.stimulus_function verbatim. `cdpk_table()` lists the default CDPK decoder panel.
"""
import dataclasses
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from biophysical_model import CalciumDynamicsModel, ModelParameters  # noqa: E402
from decoder_model import CDPKLibrary  # noqa: E402

# Comments of stimulus_function, copied from src/biophysical_model.py
STIMULI = {
    "ABA": "ABA increases PM Ca2+ influx and PLC activity (60 s latency): GLR 0.5, PLC 1.0 for t > 60 s.",
    "NaCl": "NaCl causes rapid PM channel opening: GLR 0.8, CNGC 0.6 for t > 10 s.",
    "mechanical_touch": "Touch causes very rapid, transient activation: GLR 2.0 for 0.5 < t < 5 s.",
    "flg22": "PAMP causes biphasic response: first phase PM channels (GLR 1.0, 5 < t < 60 s), "
             "second phase PLC activation (PLC 0.8, t > 30 s).",
    "cold_shock": "Cold shock rapid transient: GLR 1.2 for 2 < t < 60 s.",
    "H2O2": "H2O2 causes oscillatory PLC activation: PLC 0.5 + 0.5 sin(2π · 0.4 · (t − 30) / 60), "
            "GLR 0.3, for t > 30 s.",
    "glutamate": "Glutamate directly activates GLRs: rapid, saturating activation GLR 1 − exp(−(t − 1)/5) "
                 "for t > 1 s.",
}
STATE = [
    {"key": "Ca_cyt", "label": "Cytosolic Ca²⁺", "unit": "nM", "y0": 100.0, "note": "resting level"},
    {"key": "Ca_ER", "label": "ER Ca²⁺", "unit": "nM", "y0": 50000.0, "note": "50 μM"},
    {"key": "Ca_vac", "label": "Vacuolar Ca²⁺", "unit": "nM", "y0": 300000.0, "note": "~300 μM"},
    {"key": "IP3", "label": "InsP3", "unit": "nM", "y0": 10.0, "note": "basal"},
]


def _source_comments():
    """Map field name -> (group header, inline comment) from the ModelParameters source."""
    text = (SRC / "biophysical_model.py").read_text(encoding="utf-8")
    body = text[text.index("class ModelParameters"):text.index("class CalciumDynamicsModel")]
    group, out = "", {}
    for line in body.splitlines():
        h = re.match(r"\s+# (.+)$", line)
        f = re.match(r"\s+(\w+): float = [-\d.eE+]+\s*(?:#\s*(.*))?$", line)
        if h:
            group = h.group(1).strip()
        elif f:
            out[f.group(1)] = (group, (f.group(2) or "").strip())
    return out


def _unit(group, comment):
    for s in (comment, group):
        m = re.search(r"\(([^()]*(?:nM|/s|relative)[^()]*)\)", s)
        if m:
            return m.group(1)
    return ""


def model_parameters():
    comments = _source_comments()
    derivs = (SRC / "biophysical_model.py").read_text(encoding="utf-8")
    derivs = derivs[derivs.index("def derivatives"):derivs.index("VALID_STIMULI")]
    rows = []
    for f in dataclasses.fields(ModelParameters):
        group, comment = comments.get(f.name, ("", ""))
        rows.append({"name": f.name, "default": f.default, "group": group, "comment": comment,
                     "unit": _unit(group, comment), "used": f"p.{f.name}" in derivs})
    return rows


def cdpk_table():
    return [dataclasses.asdict(p) for p in CDPKLibrary.get_default_cdpks()]


def model_info():
    return {"parameters": model_parameters(), "stimuli": [{"name": s, "description": STIMULI[s]}
                                                         for s in CalciumDynamicsModel.VALID_STIMULI],
            "state": STATE, "cdpks": cdpk_table(),
            "defaults": {"duration": 600.0, "dt": 0.1, "intensity": 1.0}}
