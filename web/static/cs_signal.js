// Trace metrics of the repo in JavaScript: CalciumDynamicsModel.extract_signature_features
// (src/biophysical_model.py) and utils.estimate_bandwidth / estimate_snr / detect_oscillations (src/utils.py), with the
// numpy / scipy pieces they use (median, polyfit degree 1, rfft power, scipy.signal.find_peaks with height and
// distance). Tested against the Python in cs_engine.test.mjs.

export function median(a) {
  const n = a.length;
  if (!n) return NaN;
  const s = Float64Array.from(a).sort();
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
export const mean = a => { let s = 0; for (const v of a) s += v; return s / a.length; };
export function std(a) { // population SD (np.std)
  const m = mean(a); let s = 0; for (const v of a) s += (v - m) * (v - m); return Math.sqrt(s / a.length);
}
const argmax = a => { let k = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; };

// np.polyfit(x, y, 1)[0]: least-squares slope
export function slope(x, y) {
  const mx = mean(x), my = mean(y);
  let sxy = 0, sxx = 0;
  for (let i = 0; i < x.length; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) * (x[i] - mx); }
  return sxy / sxx;
}

// scipy.signal.find_peaks(x, height=h, distance=d): local maxima (plateau midpoints), height filter, then distance
// filter keeping higher peaks first.
export function findPeaks(x, height, distance) {
  const peaks = [];
  let i = 1;
  const iMax = x.length - 1;
  while (i < iMax) {
    if (x[i - 1] < x[i]) {
      let ia = i + 1;
      while (ia < iMax && x[ia] === x[i]) ia++;
      if (x[ia] < x[i]) { peaks.push(Math.floor((i + ia - 1) / 2)); i = ia; }
    }
    i++;
  }
  let p = peaks.filter(k => x[k] >= height);
  if (distance > 1 && p.length > 1) {
    const d = Math.ceil(distance), keep = p.map(() => true);
    const order = p.map((_, j) => j).sort((a, b) => x[p[a]] - x[p[b]] || a - b);
    for (let r = order.length - 1; r >= 0; r--) {
      const j = order[r];
      if (!keep[j]) continue;
      for (let k = j - 1; k >= 0 && p[j] - p[k] < d; k--) keep[k] = false;
      for (let k = j + 1; k < p.length && p[k] - p[j] < d; k++) keep[k] = false;
    }
    p = p.filter((_, j) => keep[j]);
  }
  return p;
}

const minDistance = t => Math.max(1, Math.round(5.0 / (t[1] - t[0])));
const firstAbove = (a, v) => { for (let i = 0; i < a.length; i++) if (a[i] > v) return i; return -1; };

// CalciumDynamicsModel.extract_signature_features(t, ca_trace)
export function extractFeatures(t, ca) {
  const n = ca.length;
  const baseline = median(ca.slice(0, Math.trunc(n * 0.1)));
  let peak = -Infinity; for (const v of ca) if (v > peak) peak = v;
  const threshold = baseline + 0.5 * (peak - baseline);
  const above = Array.from(ca, v => v > threshold);
  const nAbove = above.reduce((s, b) => s + b, 0);
  if (!nAbove) {
    return { baseline_ca: baseline, peak_amplitude: peak, rise_time: NaN, decay_tau: NaN, duration: 0.0,
      latency: NaN, oscillatory: false, frequency: null, n_peaks: 0 };
  }
  const latency = t[above.indexOf(true)];
  const i10 = firstAbove(ca, baseline + 0.1 * (peak - baseline));
  const i90 = firstAbove(ca, baseline + 0.9 * (peak - baseline));
  const rise_time = i10 >= 0 && i90 >= 0 ? t[i90] - t[i10] : NaN;
  const duration = nAbove * (t[1] - t[0]);
  let decay_tau = NaN;
  const pk = argmax(ca);
  if (pk < n - 10) {
    const dx = [], dy = [];
    let mx = -Infinity;
    for (let i = pk; i < n; i++) mx = Math.max(mx, ca[i] - baseline);
    if (n - pk > 5 && mx > 0) {
      for (let i = pk; i < n; i++) {
        const lv = Math.log(ca[i] - baseline + 1e-10);
        if (Number.isFinite(lv)) { dx.push(t[i] - t[pk]); dy.push(lv); }
      }
      if (dx.length > 5) { const s = slope(dx, dy); decay_tau = s !== 0 ? -1.0 / s : NaN; }
    }
  }
  const pks = findPeaks(ca, threshold, minDistance(t));
  const oscillatory = pks.length > 1;
  let frequency = null;
  if (oscillatory && pks.length > 2) frequency = 1.0 / ((t[pks[pks.length - 1]] - t[pks[0]]) / (pks.length - 1));
  return { baseline_ca: baseline, peak_amplitude: peak, rise_time, decay_tau, duration, latency, oscillatory,
    frequency, n_peaks: pks.length };
}

// utils.estimate_bandwidth: frequency below which 95% of the (mean-removed) rfft power lies
export function estimateBandwidth(t, x) {
  const n = x.length;
  let dt = 0; for (let i = 1; i < n; i++) dt += t[i] - t[i - 1]; dt /= n - 1;
  const m = mean(x), xs = Float64Array.from(x, v => v - m);
  const nf = Math.floor(n / 2) + 1, cosT = new Float64Array(n), sinT = new Float64Array(n);
  for (let k = 0; k < n; k++) { cosT[k] = Math.cos(2 * Math.PI * k / n); sinT[k] = Math.sin(2 * Math.PI * k / n); }
  const power = new Float64Array(nf);
  for (let f = 0; f < nf; f++) {
    let re = 0, im = 0, idx = 0;
    for (let j = 0; j < n; j++) { re += xs[j] * cosT[idx]; im -= xs[j] * sinT[idx]; idx += f; if (idx >= n) idx -= n; }
    power[f] = re * re + im * im;
  }
  let total = 0; for (const v of power) total += v;
  let cum = 0;
  for (let f = 0; f < nf; f++) { cum += power[f]; if (cum / total >= 0.95) return f * (1.0 / (n * dt)); }
  return (nf - 1) * (1.0 / (n * dt));
}

// utils.estimate_snr: (max - baseline mean) / baseline SD over the first 10% (amplitude ratio)
export function estimateSnr(x, baselineFraction = 0.1) {
  const b = x.slice(0, Math.trunc(x.length * baselineFraction));
  const noise = std(b);
  let mx = -Infinity; for (const v of x) if (v > mx) mx = v;
  const amp = mx - mean(b);
  return noise > 0 ? amp / noise : Infinity;
}

// utils.detect_oscillations: > 2 peaks above baseline + threshold * (max - baseline), at least 5 s apart
export function detectOscillations(t, x, threshold = 0.5) {
  const baseline = median(x.slice(0, Math.trunc(x.length * 0.1)));
  let mx = -Infinity; for (const v of x) if (v > mx) mx = v;
  const pks = findPeaks(x, baseline + threshold * (mx - baseline), minDistance(t));
  const osc = pks.length > 2;
  return [osc, osc ? 1.0 / ((t[pks[pks.length - 1]] - t[pks[0]]) / (pks.length - 1)) : null];
}
