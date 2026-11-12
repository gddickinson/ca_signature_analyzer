"""Web interface for Ca2+ Signature Analyzer (FastAPI + static ES-module front end).

- app.py         FastAPI routes under /api and STATIC_ENDPOINTS for the static GitHub Pages export
- analysis.py    trace CSV parsing and per-trace analysis (signature features, bandwidth, SNR, entropy, MI)
- signatures.py  signature-table analyses (example database, simulated demo set, capacity, classification)
- model_spec.py  model parameters / stimuli / CDPK panel read from the repo's source
- static/        front end; cs_*.js is the JS port tested against the Python (cs_engine.test.mjs)
"""
