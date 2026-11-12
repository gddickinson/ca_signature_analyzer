// The web analyses of web/analysis.py and web/signatures.py in JavaScript (static site): trace CSV parsing, per-trace
// metrics, the binned mutual-information matrix, and signature-table parsing / summary / discrimination.
// Output has the same shape as the Python API; cs_engine.test.mjs checks it against tests/fixtures/cs_fixture.json.
import { extractFeatures, estimateBandwidth, estimateSnr, detectOscillations } from './cs_signal.js';
import { entropyDiscrete, binUniform, mutualInformationContinuous, channelCapacityShannon, CDPKS, hill,
  dynamicActivation, trapezoid,
  pairwiseDiscrimination, featureVector, summaryStatistics, FEATURE_NAMES } from './cs_info.js';

export const MIN_ROWS = 20;
export const UNIT_FACTOR = { nM: 1.0, uM: 1000.0 };

// NaN / Infinity -> null, like web.analysis.clean
export function clean(v) {
  if (Array.isArray(v) || ArrayBuffer.isView(v)) return Array.from(v, clean);
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, clean(x)]));
  if (typeof v === 'number' && !Number.isFinite(v)) return null;
  return v;
}

const num = s => { s = s.trim(); if (s === '' || !/^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(s)) return null; return Number(s); };
const strip = c => c.trim().replace(/^"|"$/g, '');

// web.analysis.parse_trace_csv
export function parseTraceCsv(text) {
  const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n')
    .filter(l => l.trim() && !l.trimStart().startsWith('#'));
  if (!lines.length) throw new Error('The file is empty.');
  const first = lines[0];
  const delim = first.includes(',') ? ',' : first.includes('\t') ? '\t' : first.includes(';') ? ';' : ',';
  const head = first.split(delim).map(strip);
  let names;
  if (head.every(c => num(c) !== null)) names = ['time', ...head.slice(1).map((_, i) => `trace_${i + 1}`)];
  else { names = head; lines.shift(); }
  if (names.length < 2) throw new Error('Need at least two columns: time (s) and one [Ca2+] trace.');
  const rows = []; let dropped = 0;
  for (const l of lines) {
    const cells = l.split(delim).map(c => num(strip(c)));
    if (cells.length < names.length || cells.slice(0, names.length).some(c => c === null)) { dropped++; continue; }
    rows.push(cells.slice(0, names.length));
  }
  if (rows.length < MIN_ROWS) throw new Error(`Need at least ${MIN_ROWS} complete rows; found ${rows.length}.`);
  const t = rows.map(r => r[0]);
  const dts = t.slice(1).map((v, i) => v - t[i]);
  if (dts.some(d => d <= 0)) throw new Error('The time column (first column) must increase strictly.');
  const sorted = [...dts].sort((a, b) => a - b), m = sorted.length;
  const med = m % 2 ? sorted[(m - 1) / 2] : (sorted[m / 2 - 1] + sorted[m / 2]) / 2;
  const uniform = Math.max(...dts.map(d => Math.abs(d - med))) <= 1e-3 * med;
  const cols = {};
  for (let j = 1; j < names.length; j++) cols[names[j] || `trace_${j}`] = rows.map(r => r[j]);
  return { t, cols, info: { rows: rows.length, dropped, dt: t[1] - t[0], uniform, time_column: names[0] } };
}

// web.analysis.analyse_column
export function analyseColumn(t, ca, nBins = 10) {
  const features = extractFeatures(t, ca);
  const bandwidth = estimateBandwidth(t, ca), snr = estimateSnr(ca);
  const [osc, freq] = detectOscillations(t, ca);
  const snrPower = snr * snr;
  return {
    features, bandwidth_hz: bandwidth, snr_amplitude: snr, snr_power: snrPower,
    capacity_bits_per_s: channelCapacityShannon(bandwidth, snrPower),
    oscillation: { oscillatory: osc, frequency: freq },
    entropy_bits: entropyDiscrete(binUniform(ca, nBins)), max_entropy_bits: Math.log2(nBins),
    cdpk: CDPKS.map(c => { const a = dynamicActivation(t, ca, c); let mx = -Infinity, sm = 0; for (const v of a) { mx = Math.max(mx, v); sm += v; }
      return { name: c.name, K_d: c.K_d, n_Hill: c.n_Hill, k_off: c.k_off, peak_activation: mx, mean_activation: sm / a.length,
      integrated_activation: trapezoid(a, t), at_baseline: hill(features.baseline_ca, c.K_d, c.n_Hill), at_peak: hill(features.peak_amplitude, c.K_d, c.n_Hill) }; }),
  };
}

// web.analysis.analyse_traces
export function analyseTraces(t, cols, nBins = 10, units = 'nM') {
  if (!(units in UNIT_FACTOR)) throw new Error(`units must be one of ${Object.keys(UNIT_FACTOR).join(', ')}`);
  nBins = Math.trunc(Number(nBins));
  if (!(nBins >= 2 && nBins <= 100)) throw new Error('n_bins must be between 2 and 100.');
  const f = UNIT_FACTOR[units], names = Object.keys(cols);
  const data = Object.fromEntries(names.map(k => [k, Array.from(cols[k], v => v * f)]));
  const columns = names.map(k => ({ name: k, ...analyseColumn(t, data[k], nBins) }));
  const matrix = names.map(() => names.map(() => 0));
  names.forEach((a, i) => names.forEach((b, j) => {
    if (j >= i) matrix[i][j] = matrix[j][i] = mutualInformationContinuous(data[a], data[b], nBins);
  }));
  return clean({ n_bins: nBins, units, columns, mi: { names, matrix } });
}

// web.analysis.analyse_csv
export function analyseCsv(text, nBins = 10, units = 'nM') {
  const { t, cols, info } = parseTraceCsv(text);
  const res = analyseTraces(t, cols, nBins, units);
  res.info = info; res.t = t;
  res.traces = Object.fromEntries(Object.entries(cols).map(([k, v]) => [k, v.map(x => x * UNIT_FACTOR[units])]));
  if (!info.uniform) res.warnings = ['Time steps are not uniform: features use t[1] - t[0] as the sampling interval.'];
  return res;
}

// --- signature tables (SignatureDatabase CSV format) ---------------------------------------------------------
export function csvRows(text) { // RFC-4180-ish: quoted fields may hold commas, quotes ("") and newlines
  const rows = []; let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (ch === '"') q = false; else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === ',') { row.push(cell); cell = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); rows.push(row); row = []; cell = '';
    } else cell += ch;
  }
  if (cell !== '' || row.length) { row.push(cell); rows.push(row); }
  return rows.filter(r => r.some(c => c.trim() !== ''));
}

const REQUIRED = ['signature_id', 'species', 'cell_type', 'stimulus', 'baseline_ca', 'peak_amplitude', 'rise_time',
  'decay_tau', 'duration'];
// SignatureDatabase.load_from_csv: empty cells -> null (latency 5 s and oscillatory false by default)
export function parseSignatureCsv(text) {
  const rows = csvRows(text);
  if (rows.length < 2) throw new Error('No signatures in the file.');
  const head = rows[0].map(h => h.trim());
  const missing = REQUIRED.filter(c => !head.includes(c));
  if (missing.length) throw new Error(`Could not read the table: missing column(s) ${missing.join(', ')}`);
  const get = (r, k) => { const i = head.indexOf(k); const v = i < 0 ? '' : (r[i] ?? '').trim(); return v === '' ? null : v; };
  const numOr = (r, k, d = null) => { const v = get(r, k); return v === null ? d : Number(v); };
  return rows.slice(1).map(r => {
    const o = get(r, 'oscillatory');
    return {
      signature_id: get(r, 'signature_id'), species: get(r, 'species'), cell_type: get(r, 'cell_type'),
      stimulus: get(r, 'stimulus'), stimulus_concentration: numOr(r, 'stimulus_concentration'),
      baseline_ca: numOr(r, 'baseline_ca', NaN), peak_amplitude: numOr(r, 'peak_amplitude', NaN),
      rise_time: numOr(r, 'rise_time', NaN), decay_tau: numOr(r, 'decay_tau', NaN), duration: numOr(r, 'duration', NaN),
      latency: numOr(r, 'latency', 5.0), oscillatory: o === null ? false : ['true', '1', 'yes', '1.0'].includes(o.toLowerCase()),
      frequency: numOr(r, 'frequency'), duty_cycle: numOr(r, 'duty_cycle'),
      spike_amplitude_cv: numOr(r, 'spike_amplitude_cv'), wave_velocity: numOr(r, 'wave_velocity'),
      spatial_range: numOr(r, 'spatial_range'), reference: get(r, 'reference'), notes: get(r, 'notes'),
    };
  });
}

// web.signatures.analyse_table(db, full=False): the parts with a JS port
export function analyseTable(sigs) {
  const X = sigs.map(featureVector), labels = sigs.map(s => s.stimulus);
  const classes = [...new Set(labels)].sort();
  const rows = sigs.map(s => ({ ...s, fold_change: s.peak_amplitude / s.baseline_ca,
    snr: s.baseline_ca * 0.1 > 0 ? (s.peak_amplitude - s.baseline_ca) / (s.baseline_ca * 0.1) : Infinity }));
  return clean({ rows, feature_names: FEATURE_NAMES, features: X, summary: summaryStatistics(sigs), classes,
    class_counts: classes.map(c => labels.filter(l => l === c).length), discrimination: pairwiseDiscrimination(X, labels) });
}
