// Labels, units and table rows for the signature features of CalciumDynamicsModel.extract_signature_features.
import { esc } from './core.js';
import { fmt } from './plots.js';

export const FEATURES = [
  ['baseline_ca', 'Baseline', 'nM', 'median of the first 10% of the trace'],
  ['peak_amplitude', 'Peak', 'nM', 'maximum of the trace'],
  ['rise_time', 'Rise time', 's', '10% to 90% of (peak − baseline), first crossings'],
  ['decay_tau', 'Decay τ', 's', 'exponential fit (log-linear) from the peak to the end'],
  ['duration', 'Duration', 's', 'total time above half-maximum'],
  ['latency', 'Latency', 's', 'time (from t = 0) of the first half-maximum crossing'],
  ['oscillatory', 'Oscillatory', '', 'more than one peak above half-maximum, ≥ 5 s apart'],
  ['n_peaks', 'Peaks', '', 'peaks above half-maximum, ≥ 5 s apart'],
  ['frequency', 'Frequency', 'Hz', '1 / mean inter-peak interval (needs > 2 peaks)'],
];

export const featureRows = f => FEATURES.map(([k, label, unit, note]) =>
  `<tr><th>${esc(label)}</th><td class="num">${fmt(f[k])}${unit && f[k] !== null && typeof f[k] === 'number' ? ' ' + unit : ''}</td><td class="small mute">${esc(note)}</td></tr>`).join('');
