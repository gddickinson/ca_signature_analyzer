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

### Needs George's decision (open)
- **Pump capacity**: with Ca2+ conserved, NaCl and glutamate at intensity 1 (and ABA above ~1.3) bring in more Ca2+
  than the PM Ca-ATPase can remove (V_PM_pump 40 nM/s; GLR + CNGC influx plus GLR-linked vacuolar release), so
  cytosolic Ca2+ rises without a steady state (14-19 uM at 600 s). A calibration question (V_PM_pump,
  V_vac_release, stimulus gains).
- **Feature definitions**: the baseline (median of the first 10%) includes the response for fast stimuli; latency is
  measured from t = 0; decay tau fits the whole post-peak trace. Documented on the web Format & metrics page.
- The literature references of the example table were not re-checked.
### Notes
- The shared venv lacks umap-learn; `analyze_database.py` was checked end to end with a PCA stand-in for UMAP.
- Not pushed; the lead reviews and merges.

## 2026-11-11 (later): model and statistics fixes requested by George (changes results by design)
Each a "Fix ..." commit with regression tests that failed first; JS port and fixture updated; KIT checks rerun.
1. Ca2+ conservation: fluxes in cytosolic-concentration units, stores change by -J_net / vol_ratio
   (De Young & Keizer 1992; Li & Rinzel 1994); passive leaks V_PM_leak 8 nM/s, k_ER_leak 2.4e-4 /s,
   k_vac_leak 3.33e-5 /s give a resting state near 100 nM / 50 uM / 300 uM. ABA (I = 1) peaks at 623 nM (was 72 uM).
2. CDPK kinetics: da/dt = k_off (([Ca]/K_d)^n (1 - a) - a), exact step update (was forward Euler with k_on in
   M^-1 s^-1 used as nM^-1 s^-1, flipping 0/1); k_on relabelled M^-1 s^-1 and unused (library k_on, k_off and K_d are
   mutually inconsistent; K_d taken as authoritative). PhosphorylationTarget also integrated exactly. The web trace
   analysis shows dynamic activation (peak, integral).
3. Example frequencies (ABA 0.5, flg22 0.2, H2O2 0.4) were per minute; stored as Hz (/60) in code and CSV.
4. Information flow initial entropy binned like the MI; preservation printed as a ratio; classification information
   = MI of the confusion matrix (plug-in and Miller-Madow); `bias_correction='miller_madow'` for MI and
   `entropy_miller_madow`; the web shows both estimates.
