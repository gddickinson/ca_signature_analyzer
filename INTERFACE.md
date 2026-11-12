# Ca2+ Signature Analyzer -- Interface Map

## Package: `src/`

### `biophysical_model.py`
- `ModelParameters` -- ODE model parameters (channel conductances, pump activities, etc.)
- `CalciumDynamicsModel` -- ODE-based Ca2+ dynamics simulation (Ca2+-conserving store fluxes, passive leaks)
  - `simulate(stimulus_type, duration, dt, intensity)` -- run simulation (with input validation)
  - `extract_signature_features(t, ca_trace)` -- extract peak, rise time, decay tau, etc.
  - `VALID_STIMULI` -- class-level tuple of accepted stimulus names

### `decoder_model.py`
- `CDPKParameters` -- CDPK kinetic parameters (Kd, Hill coeff, rates)
- `CDPKLibrary` -- default library of 10 Arabidopsis CDPKs
- `CDPKDecoder` -- single CDPK activation model (Hill steady state + Hill-type kinetics, exact step update)
- `DecoderPanel` -- panel of multiple CDPKs; `analyze_signature()`, `get_response_matrix()`
- `PhosphorylationTarget` -- target protein phosphorylation dynamics

### `information_theory.py`
- `InformationAnalyzer` -- MI (discrete/continuous, optional `bias_correction='miller_madow'`), entropy, `entropy_miller_madow`, channel capacity, signature capacity
- `PathwayInformationFlow` -- information flow between signaling stages (text label stages become integer codes)
- re-exports `SignatureDiscriminability` from `discriminability.py`

### `discriminability.py`
- `SignatureDiscriminability` -- pairwise discrimination (standardized class-mean distance), classification accuracy (information = MI of the confusion matrix)

### `signature_database.py`
- `CalciumSignature` -- single signature record
- `SignatureDatabase` -- collection with load/save CSV, feature matrix, summary stats

### `visualization.py`
- `SignaturePlotter` -- static plotting methods (heatmaps, UMAP, comparisons)
- `IntegratedReport` -- report generation

### `utils.py`
- Utility functions

### GUI (split across 3 files):
- `gui_app.py` -- main GUI class `CaSignatureAnalyzerGUI` (uses mixins), entry point `main()`
- `gui_tabs.py` -- tab construction mixins (Database, Simulation, Decoder, Information, Visualization)
- `gui_callbacks.py` -- callback method mixins for all GUI actions

## Entry Points
- `launch_gui.py` -- launches the GUI
- `run_simulation.py` -- CLI simulation runner
- `analyze_database.py` -- CLI database analysis (`encode_stimuli()` gives sorted stimulus codes)
- `run_web.py` -- starts the web interface (FastAPI, http://127.0.0.1:8134)

## Web interface: `web/` (imports `src/` flat, like the scripts)
- `app.py` -- FastAPI routes: GET /api/info, /api/model, /api/database, /api/demo_signatures (exported to JSON for the
  static site via `STATIC_ENDPOINTS`); POST /api/simulate, /api/analyze, /api/signatures
- `analysis.py` -- `parse_trace_csv`, `analyse_column`, `analyse_traces`, `analyse_csv`, `simulate`, `clean` (JSON-safe)
- `signatures.py` -- `example_database`, `simulated_set` (7 stimuli x 5 intensities), `analyse_table`, `python_only`
- `model_spec.py` -- `model_parameters()` (defaults/units/comments parsed from biophysical_model.py), `STIMULI`, `cdpk_table`
- `static/` -- ES-module front end (no build step):
  - `index.html`, `style.css` (shared kit theme), `style_app.css`, `core.js` (kit router/helpers), `main.js`, `icon.svg`, `background.js`
  - pages: `home.js`, `model.js`, `analyse.js` + `analyse_view.js`, `signatures.js`, `metrics.js`, `about.js`;
    helpers `plots.js` (SVG line plots, heatmaps, bars), `feature_table.js`
  - JS port: `cs_model.js` (ODE model, Dormand-Prince integrator), `cs_signal.js` (features, bandwidth, SNR, find_peaks),
    `cs_info.js` (entropy, MI, binning, discrimination, CDPK Hill), `cs_analysis.js` (mirrors web/analysis.py + signature tables)
  - `static_api.js` -- answers api() on the static site with the JS port; `cs_engine.test.mjs` -- node test vs Python

## Scripts: `scripts/`
- `make_cs_fixture.py` -- writes `tests/fixtures/cs_fixture.json` (Python results the JS port is tested against)
- `build_static.py` -- exports the static site to `site/`; `screenshots.py` -- Playwright smoke test + README images

## Tests: `tests/`
- `test_biophysical_model.py` -- ODE model and feature extraction tests
- `test_information_theory.py` -- MI, entropy, classification tests
- `test_decoder_model.py` -- CDPK activation and decoder panel tests
- `test_signature_database.py` -- CSV loading (empty cells, round trip)
- `test_web.py` -- web API, static build, fixture currency, node test of the JS port
