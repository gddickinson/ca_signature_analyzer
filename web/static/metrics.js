// Format & metrics: the input formats and what each number means, as the repo's code computes it.
import { route, app, download } from './core.js';
import { demoCsv } from './analyse.js';

const D = (term, body) => `<dt>${term}</dt><dd>${body}</dd>`;

route('metrics', () => {
  app().innerHTML = `<h1 class="glow">Format &amp; metrics</h1>
  <p class="lead">What the analyses read and what each number means. Definitions follow the repo's code
    (<code>src/biophysical_model.py</code>, <code>src/utils.py</code>, <code>src/information_theory.py</code>,
    <code>src/discriminability.py</code>, <code>src/decoder_model.py</code>, <code>src/signature_database.py</code>).</p>

  <h2>Trace CSV (Analyse traces)</h2>
  <ul>
    <li>First column: <b>time in seconds</b>, strictly increasing and evenly sampled (the features use t[1] − t[0] as the
      sampling interval; uneven sampling gives a warning).</li>
    <li>Every further column: one <b>[Ca²⁺] trace</b>, in nM (or choose μM, which is multiplied by 1000). The column header is the trace name.</li>
    <li>Comma-separated; tab or semicolon also work. A header row is optional; lines starting with <code>#</code> are ignored;
      rows with an empty or non-numeric cell are skipped. At least 20 complete rows.</li>
    <li>Ratiometric or ΔF/F data can be analysed too, but then concentrations (baseline, peak) are in those units and the CDPK
      activation, which needs nM, does not apply.</li>
  </ul>
  <pre>time_s,cell_1,cell_2
0.0,101.2,98.7
0.1,100.8,99.1
...</pre>
  <button id="demo" class="alt">Download the simulated demo CSV</button>

  <h2>Signature table CSV (Signature tables)</h2>
  <p>The format of <code>SignatureDatabase.save_to_csv</code> / <code>load_from_csv</code> (as <code>data/example_signatures.csv</code>).
    Required columns: <code>signature_id, species, cell_type, stimulus, baseline_ca, peak_amplitude, rise_time, decay_tau, duration</code>.
    Optional: <code>stimulus_concentration</code> (M; the example stores 10 μM as 1e-05), <code>latency</code> (s, default 5),
    <code>oscillatory</code> (True/False), <code>frequency</code> (Hz), <code>duty_cycle</code>, <code>spike_amplitude_cv</code>,
    <code>wave_velocity</code> (μm/s), <code>spatial_range</code> (μm), <code>reference</code>, <code>notes</code>.
    Concentrations in nM, times in s. Empty cells are allowed in optional columns.</p>

  <h2>Signature features</h2>
  <p class="small mute">CalciumDynamicsModel.extract_signature_features(t, ca), the same function for simulated and uploaded traces.</p>
  <dl class="metrics">
    ${D('Baseline (nM)', 'Median of the first 10% of the samples. If the response starts within that window (for a 600 s trace: the first 60 s, e.g. touch at 0.5 s or glutamate at 1 s) the baseline includes part of the response.')}
    ${D('Peak (nM)', 'Maximum of the trace (absolute, not baseline-subtracted). Fold change in the signature tables is peak / baseline.')}
    ${D('Rise time (s)', 'Time from the first sample above baseline + 10% of (peak − baseline) to the first sample above baseline + 90%.')}
    ${D('Decay τ (s)', 'From the peak to the end of the trace: a straight line fitted to ln(ca − baseline) against time (points below baseline are left out); τ = −1 / slope. For a trace that does not decay to baseline this is not a decay constant (it can be very large or negative).')}
    ${D('Duration (s)', 'Number of samples above half-maximum (baseline + 0.5 (peak − baseline)) × sampling interval: the total time above half-maximum, summed over all excursions.')}
    ${D('Latency (s)', 'Time of the first sample above half-maximum, measured from t = 0 of the trace (not from stimulus onset).')}
    ${D('Peaks, oscillatory, frequency', 'scipy.signal.find_peaks with height = half-maximum and a minimum spacing of 5 s. Oscillatory if more than one peak; frequency (Hz) = 1 / mean inter-peak interval when there are more than two peaks. utils.detect_oscillations uses the stricter rule (more than two peaks).')}
  </dl>

  <h2>Signal and information measures (per trace)</h2>
  <dl class="metrics">
    ${D('Bandwidth (Hz)', 'utils.estimate_bandwidth: power spectrum (|FFT|²) of the mean-subtracted trace; the frequency below which 95% of the power lies.')}
    ${D('SNR (amplitude)', 'utils.estimate_snr: (maximum − mean of the first 10%) / standard deviation of the first 10%. An amplitude ratio, not a power ratio; infinite (shown as –) for a noise-free flat baseline.')}
    ${D('Shannon capacity (bit/s)', '<span class="formula">C = W · log₂(1 + SNR_power)</span> (InformationAnalyzer.channel_capacity_shannon, Shannon–Hartley). This page uses W = the 95%-power bandwidth and SNR_power = SNR² (the theorem needs a power ratio). The repo does not combine these two estimates itself; treat the result as an order-of-magnitude bound for a Gaussian channel.')}
    ${D('Amplitude entropy (bit)', '<span class="formula">H = −Σ p log₂ p</span> over the trace values binned into N equal-width bins between their minimum and maximum (KBinsDiscretizer, strategy "uniform", then InformationAnalyzer.entropy_discrete). At most log₂ N. It is the plug-in estimate, treating every time point as an independent sample.')}
    ${D('Mutual information (bit)', '<span class="formula">I(X;Y) = H(X) + H(Y) − H(X,Y)</span> between two traces sampled at the same times, each binned into N equal-width bins (InformationAnalyzer.mutual_information_continuous). Plug-in estimate: biased upwards when there are few samples per occupied bin pair, and neighbouring time points are not independent, so compare values between traces of the same length and bin number rather than reading them as absolute.')}
    ${D('CDPK activation', 'CDPKDecoder.steady_state_activation, the Hill equation <span class="formula">[Ca]ⁿ / (K_dⁿ + [Ca]ⁿ)</span> for the default panel of ten Arabidopsis CDPKs (K_d 100–2000 nM, n 2–3.5), evaluated at the baseline and the peak.')}
  </dl>

  <h2>Signature-table measures</h2>
  <dl class="metrics">
    ${D('Feature vector', 'CalciumSignature.get_feature_vector: baseline, peak, fold change, rise time, decay τ, duration, latency, frequency (0 if missing), duty cycle (0 if missing).')}
    ${D('Pairwise discrimination', 'SignatureDiscriminability.pairwise_discrimination: <span class="formula">‖(μ₁ − μ₂) / ((σ₁ + σ₂)/2)‖</span> over the nine features, with σ the population standard deviation of each stimulus class (+1e-10). Needs several signatures per stimulus.')}
    ${D('Mutual information per feature', 'InformationAnalyzer.estimate_signature_capacity: scikit-learn mutual_info_classif (k-nearest-neighbour estimator, 3 neighbours) between each feature and the stimulus label, in bits; maximum log₂(number of stimuli); efficiency = mean over features / maximum. Python only.')}
    ${D('Classification accuracy', 'SignatureDiscriminability.classification_accuracy: Random Forest (100 trees, random_state 42), 5-fold cross-validated accuracy (mean ± SD over folds) and the confusion matrix. Its "information transmitted" is log₂(N) × accuracy, a simplification stated in the code, not a mutual-information estimate. Python only.')}
  </dl>

  <h2>The biophysical model</h2>
  <p>CalciumDynamicsModel.derivatives, all concentrations in nM, time in s; stimuli set the GLR, CNGC and PLC activities.
    Fluxes J are in cytosolic-concentration units, so a store of relative volume v changes by J / v and store exchange
    conserves total Ca²⁺ (Ca_cyt + v_ER·Ca_ER + v_vac·Ca_vac), as in De Young &amp; Keizer (1992) and Li &amp; Rinzel (1994).
    Passive leaks balance the pumps at rest (Ca_cyt ≈ 100 nM).</p>
  <pre>J_PM_in       = V_PM_GLR·GLR + V_PM_CNGC·CNGC
J_ER_release  = V_ER_release · IP3ⁿ/(K_IP3ⁿ + IP3ⁿ) · Ca_ER/Ca_ER_max
J_vac_release = V_vac_release · Ca_vac/(K_vac + Ca_vac) · GLR
J_pump,x      = V_x_pump · Ca_cyt²/(K_pump² + Ca_cyt²)          (x = PM, ER, vac)
J_leak        = V_PM_leak;  k_ER_leak·(Ca_ER − Ca_cyt);  k_vac_leak·(Ca_vac − Ca_cyt)
J_ER_net  = J_ER_release + J_ER_leak − J_ER_pump      J_vac_net = J_vac_release + J_vac_leak − J_vac_pump
dCa_cyt/dt = J_PM_in + V_PM_leak − J_PM_pump + J_ER_net + J_vac_net
dCa_ER/dt  = −J_ER_net / vol_ratio_ER
dCa_vac/dt = −J_vac_net / vol_ratio_vac
dIP3/dt    = V_PLC·PLC − k_IP3_deg·IP3</pre>
  <p class="small mute">Initial state: Ca_cyt 100 nM, Ca_ER 50 μM, Ca_vac 300 μM, IP3 10 nM. Buffering parameters (B_total, K_buffer)
    and K_ER, Ca_vac_max are defined but not used by these equations. With the default parameters a sustained influx
    larger than the PM Ca-ATPase maximum (V_PM_pump = 40 nM/s), as for NaCl and glutamate at intensity 1 (GLR- and
    CNGC-mediated influx plus GLR-linked vacuolar release), has no steady state: cytosolic Ca²⁺ keeps rising.</p>`;
  document.getElementById('demo').onclick = () => download('simulated_demo_traces.csv', demoCsv(), 'text/csv');
});
