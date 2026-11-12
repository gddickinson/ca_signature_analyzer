# Session log

## 2026-11-11: web interface, static site, bug fixes (Claude, branch `web-interface`)

### Added
- `web/` (FastAPI app, port 8134 via `run_web.py`) and a static GitHub Pages build (`scripts/build_static.py`,
  `.github/workflows/pages.yml`). Pages: Model explorer, Analyse traces, Signature tables, Format & metrics, About.
- JS port of the model and analyses (`web/static/cs_*.js`), node-tested against Python
  (`scripts/make_cs_fixture.py` -> `tests/fixtures/cs_fixture.json`, `web/static/cs_engine.test.mjs`):
  derivatives 1e-12; trajectories vs odeint(rtol 1e-12) < 1e-7 (worst 4.9e-8), vs the repo's `simulate()` 8.7e-7;
  features, bandwidth, SNR, entropy, MI, full CSV analyses and signature tables 1e-9.
- `tests/test_web.py` (API, static build, fixture currency, node test), `tests/test_signature_database.py`.
- `requirements-web.txt` (pinned), README section with screenshots (`docs/img/`).

### Bugs fixed (each in its own commit with a regression test that failed first)
1. `np.trapz` removed in NumPy 2: every decoder analysis crashed (`decoder_model.py`).
2. `extract_signature_features`: flat traces returned no `oscillatory`/`frequency`/`n_peaks` (GUI KeyError);
   5 s peak spacing was `int(5 // dt)` = 49 samples at dt 0.1 and 0 (scipy error) for dt > 5 s (also in utils).
3. `load_from_csv`: empty cells became NaN (9 NaNs in the example feature matrix); empty `oscillatory` became True.
4. `PathwayInformationFlow`: stimulus-name stages (as passed by the CLI and GUI) crashed KBinsDiscretizer.
5. `analyze_database.py`: crashed at step 3 on the example database (k-NN MI needs >= 2 signatures per stimulus,
   now a clear error and a skip); confusion-matrix axis labels used `list(set(...))` order instead of sorted order.
6. Docs only: buffering, K_ER, Ca_vac_max are unused by the equations; "Bhattacharyya" discrimination is a
   standardized mean distance; `estimate_snr` is an amplitude ratio (Shannon-Hartley needs power); initial ER Ca2+
   is 50 uM, not ~500 uM. `SignatureDiscriminability` moved to `src/discriminability.py` (500-line budget).
7. Untracked committed `src/__pycache__/*.pyc`.

### Needs George's decision (not changed: would change model / analysis results)
- **Model mass balance** (`CalciumDynamicsModel.derivatives`): ER release adds `J/vol_ratio_ER` (10 J) to the
  cytosol but removes `vol_ratio_ER * J` (0.1 J) from the ER; per cytosolic volume that is 10 J gained vs 0.01 J lost
  (factor 1000); the vacuole likewise (factor 100 the other way). Conservative form: cytosol +-J, store -+J/vol_ratio.
  Consequences: ABA and flg22 run away to 72 uM / 30 uM cytosolic Ca2+ in 600 s; glutamate drains the vacuole from
  300 uM to 0.14 uM; with no stimulus Ca_cyt falls from 100 nM to < 1 nM (no resting steady state).
- **CDPK dynamic activation** (`CDPKDecoder.dynamic_activation`): k_on = 1e7-1.2e8 "1/nM/s" looks like M^-1 s^-1
  values (1e8 M^-1 s^-1 = 0.1 nM^-1 s^-1); with forward Euler at dt 0.1 s the step is ~1e9, so activation alternates
  0, 1, 0, 1 (clipped). K_d and n_Hill do not enter the kinetics (implied K_d = k_off/k_on ~ 1e-6 nM). Affects
  integrated activation, the decoder heatmap and the CDPK stage of the information flow. The web shows only the
  Hill steady state.
- **Example database frequencies**: ABA 0.5 Hz, flg22 0.2 Hz, H2O2 0.4 Hz are labelled Hz; the model's H2O2
  oscillation uses 0.4/60 Hz (0.4 per minute), so the table values are probably per minute. Please check the sources.
- **PathwayInformationFlow**: `initial_entropy` uses KBinsDiscretizer's default quantile binning while the MI uses
  uniform binning (7 equiprobable stimuli give 2.52 bits instead of log2 7 = 2.81); `information_preservation` is a
  ratio printed as "bits"; a multi-feature first stage is flattened with `reshape(-1, 1)`.
- **classification_accuracy information_bits** = log2(N) x accuracy (flagged "simplified" in the code): chance
  accuracy gives log2(N)/N bits, not 0. MI from the confusion matrix would be a proper estimate.
- **Plug-in entropy / MI** with 10 x 10 bins on 7-35 signatures (information flow) is strongly biased upward; a
  bias correction (Miller-Madow) or a shuffled-label baseline would make values interpretable.
- **Features**: the baseline (median of the first 10%) includes the response for fast stimuli (touch 0.5 s,
  glutamate 1 s, cold 2 s in a 600 s trace); latency is measured from t = 0, not stimulus onset; decay tau fits the
  whole post-peak trace (negative or huge for non-decaying traces). Documented on the web Format & metrics page.

### Notes
- The shared venv lacks umap-learn; `analyze_database.py` was checked end to end with a PCA stand-in for UMAP.
- Not pushed; the lead reviews and merges.
