"""FastAPI web app for Ca2+ Signature Analyzer: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8134, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. The POST
routes are answered in the browser on the static site by static_api.js (JS port in cs_*.js), except the
Python-only estimators of POST /signatures (k-NN mutual information, Random Forest).
"""
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from web import analysis, signatures
from web.model_spec import model_info

STATIC = Path(__file__).resolve().parent / "static"
DESCRIPTION = ("A Python toolkit for analyzing plant calcium signaling dynamics using information theory and "
               "biophysical modeling.")

app = FastAPI(title="Ca2+ Signature Analyzer")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary for the home page."""
    m = model_info()
    return {"title": "Ca²⁺ Signature Analyzer", "description": DESCRIPTION,
            "n_stimuli": len(m["stimuli"]), "n_parameters": len(m["parameters"]), "n_cdpks": len(m["cdpks"]),
            "n_state": len(m["state"]), "n_examples": len(signatures.example_database())}


@app.get("/api/model")
def model():
    """Model parameters (defaults, units, comments from the source), stimuli, state variables, CDPK panel."""
    return model_info()


@app.get("/api/database")
def database():
    """The repo's example database (create_example_database), analysed."""
    return signatures.analyse_table(signatures.example_database())


@app.get("/api/demo_signatures")
def demo_signatures():
    """SIMULATED signature table from the biophysical model (7 stimuli x 5 intensities), analysed in Python."""
    return signatures.analyse_table(signatures.simulated_set())


def _bad(e):
    raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/simulate")
def simulate(body: dict = Body(...)):
    try:
        return analysis.simulate(body.get("stimulus", "ABA"), body.get("duration", 600.0), body.get("dt", 0.1),
                                 body.get("intensity", 1.0), body.get("params"))
    except (ValueError, TypeError) as e:
        _bad(e)


@app.post("/api/analyze")
def analyze(body: dict = Body(...)):
    """Trace CSV text -> per-trace metrics and the MI matrix."""
    try:
        return analysis.analyse_csv(body.get("text", ""), body.get("n_bins", 10), body.get("units", "nM"))
    except (ValueError, TypeError) as e:
        _bad(e)


@app.post("/api/signatures")
def signature_table(body: dict = Body(...)):
    """Signature-table CSV text (SignatureDatabase format) -> summary, discrimination, capacity, classification."""
    try:
        return signatures.analyse_table(signatures.load_csv_text(body.get("text", "")))
    except (ValueError, TypeError, KeyError) as e:
        _bad(f"Could not read the table: {e}")


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/model", "/database", "/demo_signatures"]
