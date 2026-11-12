// Information-theory and signature-table code of the repo in JavaScript:
// InformationAnalyzer.entropy_discrete / mutual_information_discrete / mutual_information_continuous /
// channel_capacity_shannon (src/information_theory.py; uniform KBinsDiscretizer binning, sklearn mutual_info_score),
// SignatureDiscriminability.pairwise_discrimination (src/discriminability.py), CDPKDecoder.steady_state_activation and
// the default CDPK panel (src/decoder_model.py), CalciumSignature.get_feature_vector and
// SignatureDatabase.get_summary_statistics (src/signature_database.py). Tested in cs_engine.test.mjs.

const LN2 = Math.log(2);

function counts(codes) {
  const m = new Map();
  for (const c of codes) m.set(c, (m.get(c) || 0) + 1);
  return [...m.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0)).map(e => e[1]);
}

// entropy_discrete (1-D): plug-in Shannon entropy in bits (scipy.stats.entropy(p, base=2))
export function entropyDiscrete(x) {
  const c = counts(x), n = x.length;
  const p = c.map(v => v / n);
  let tot = 0; for (const v of p) tot += v;
  let s = 0; for (const v of p) { const q = v / tot; if (q > 0) s += -q * Math.log(q); }
  return s / LN2;
}

// sklearn.metrics.mutual_info_score (nats) on integer labels, then / ln 2 (mutual_information_discrete)
export function mutualInformationDiscrete(x, y) {
  const n = x.length, xi = new Map(), yi = new Map();
  [...new Set(x)].sort((a, b) => a - b).forEach((v, i) => xi.set(v, i));
  [...new Set(y)].sort((a, b) => a - b).forEach((v, i) => yi.set(v, i));
  if (xi.size === 1 || yi.size === 1) return 0.0;
  const cont = new Map(), pi = new Array(xi.size).fill(0), pj = new Array(yi.size).fill(0);
  for (let k = 0; k < n; k++) {
    const a = xi.get(x[k]), b = yi.get(y[k]), key = a * yi.size + b;
    cont.set(key, (cont.get(key) || 0) + 1); pi[a]++; pj[b]++;
  }
  const keys = [...cont.keys()].sort((a, b) => a - b);
  const logSum = Math.log(n);
  let mi = 0;
  for (const key of keys) {
    const v = cont.get(key), a = Math.floor(key / yi.size), b = key % yi.size;
    const cnm = v / n, logOuter = -Math.log(pi[a] * pj[b]) + logSum + logSum;
    let term = cnm * (Math.log(v) - logSum) + cnm * logOuter;
    if (Math.abs(term) < Number.EPSILON) term = 0;
    mi += term;
  }
  return Math.max(mi, 0) / LN2;
}

// KBinsDiscretizer(n_bins, encode='ordinal', strategy='uniform').fit_transform(x) as integer codes
export function binUniform(x, nBins) {
  let lo = Infinity, hi = -Infinity;
  for (const v of x) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (lo === hi) return Array.from(x, () => 0);
  const step = (hi - lo) / nBins, inner = [];
  for (let i = 1; i < nBins; i++) inner.push(i * step + lo); // np.linspace edges[1:-1]
  return Array.from(x, v => { // searchsorted(inner, v, side='right')
    let a = 0, b = inner.length;
    while (a < b) { const m = (a + b) >> 1; if (inner[m] <= v) a = m + 1; else b = m; }
    return a;
  });
}

export const mutualInformationContinuous = (x, y, nBins = 10) =>
  mutualInformationDiscrete(binUniform(x, nBins), binUniform(y, nBins));

// Shannon-Hartley: C = W log2(1 + SNR), SNR a power ratio
export const channelCapacityShannon = (bandwidth, snr) => bandwidth * Math.log2(1 + snr);

// Default CDPK panel (CDPKLibrary.get_default_cdpks: name, K_d nM, Hill n) and the Hill steady state
export const CDPKS = [['CPK1', 2000, 3.0], ['CPK2', 1500, 2.8], ['CPK3', 500, 3.2], ['CPK4', 400, 3.0],
  ['CPK5', 300, 3.5], ['CPK6', 350, 3.4], ['CPK21', 150, 2.5], ['CPK23', 180, 2.6], ['CPK28', 100, 2.0],
  ['CPK32', 120, 2.2]].map(([name, K_d, n_Hill]) => ({ name, K_d, n_Hill }));
export function hill(ca, K_d, n) { const cn = ca ** n; return cn / (K_d ** n + cn); }

// SignatureDiscriminability.pairwise_discrimination: ||(mu1 - mu2) / ((sd1 + sd2) / 2)||, sd + 1e-10
export function pairwiseDiscrimination(X, labels) {
  const classes = [...new Set(labels)].sort();
  const nf = X[0].length;
  const stats = classes.map(c => {
    const rows = X.filter((_, i) => labels[i] === c);
    const mu = [], sd = [];
    for (let f = 0; f < nf; f++) {
      let m = 0; for (const r of rows) m += r[f]; m /= rows.length;
      let v = 0; for (const r of rows) v += (r[f] - m) * (r[f] - m);
      mu.push(m); sd.push(Math.sqrt(v / rows.length) + 1e-10);
    }
    return { mu, sd };
  });
  return classes.map((_, i) => classes.map((__, j) => {
    if (i === j) return 0;
    let s = 0;
    for (let f = 0; f < nf; f++) {
      const z = (stats[i].mu[f] - stats[j].mu[f]) / ((stats[i].sd[f] + stats[j].sd[f]) / 2);
      s += z * z;
    }
    return Math.sqrt(s);
  }));
}

export const FEATURE_NAMES = ['baseline_ca', 'peak_amplitude', 'fold_change', 'rise_time', 'decay_tau', 'duration',
  'latency', 'frequency', 'duty_cycle'];
// CalciumSignature.get_feature_vector
export const featureVector = s => [s.baseline_ca, s.peak_amplitude, s.peak_amplitude / s.baseline_ca, s.rise_time,
  s.decay_tau, s.duration, s.latency, s.frequency ?? 0, s.duty_cycle ?? 0];

// SignatureDatabase.get_summary_statistics
export function summaryStatistics(sigs) {
  const n = sigs.length, avg = a => a.reduce((s, v) => s + v, 0) / a.length;
  const freqs = sigs.map(s => s.frequency).filter(v => v !== null && v !== undefined && !Number.isNaN(v));
  return {
    total_signatures: n,
    unique_stimuli: new Set(sigs.map(s => s.stimulus)).size,
    unique_cell_types: new Set(sigs.map(s => s.cell_type)).size,
    oscillatory_fraction: avg(sigs.map(s => (s.oscillatory ? 1 : 0))),
    mean_peak_amplitude: avg(sigs.map(s => s.peak_amplitude)),
    mean_duration: avg(sigs.map(s => s.duration)),
    mean_frequency: freqs.length ? avg(freqs) : null,
  };
}
