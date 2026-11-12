// node cs_engine.test.mjs: the JS port (cs_model.js, cs_signal.js, cs_info.js, cs_analysis.js) against the repo's
// Python on the same inputs (tests/fixtures/cs_fixture.json, written by scripts/make_cs_fixture.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { derivatives, stimulus, simulate, DEFAULTS } from './cs_model.js';
import { extractFeatures, estimateBandwidth, estimateSnr, detectOscillations } from './cs_signal.js';
import { entropyDiscrete, entropyMillerMadow, mutualInformationDiscrete, binUniform, mutualInformationContinuous } from './cs_info.js';
import { analyseCsv, parseSignatureCsv, analyseTable } from './cs_analysis.js';

const F = JSON.parse(readFileSync(new URL('../../tests/fixtures/cs_fixture.json', import.meta.url)));
let n = 0;
const ok = (name, fn) => { fn(); n++; console.log('ok  ', name); };
const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-300);

// Deep comparison: numbers to rtol (or atol near zero), null/booleans/strings exactly, same keys.
function close(js, py, rtol, path = '$', atol = 1e-12) {
  if (py === null || typeof py !== 'object') {
    if (typeof py === 'number' && typeof js === 'number') {
      if (Math.abs(js - py) > atol && rel(js, py) > rtol) assert.fail(`${path}: JS ${js} vs Python ${py}`);
    } else assert.equal(js, py, path);
    return;
  }
  if (Array.isArray(py)) {
    assert.equal(js.length, py.length, `${path} length`);
    py.forEach((v, i) => close(js[i], v, rtol, `${path}[${i}]`, atol));
    return;
  }
  assert.deepEqual(Object.keys(js).sort(), Object.keys(py).sort(), `${path} keys`);
  for (const k of Object.keys(py)) close(js[k], py[k], rtol, `${path}.${k}`, atol);
}
const nul = v => (Number.isFinite(v) ? v : null);

ok(`stimulus_function and derivatives() equal Python at ${F.derivatives.length} sampled states (1e-12)`, () => {
  for (const c of F.derivatives) {
    const p = { ...DEFAULTS, ...c.params };
    close(stimulus(c.t, c.stimulus, c.intensity), c.stim, 1e-12);
    close(derivatives(c.y, c.t, c.stimulus, c.intensity, p), c.dy, 1e-12, '$', 1e-9);
  }
});

let worstTight = 0;
ok(`ODE trajectories match odeint at rtol 1e-12 for ${F.ode.length} runs (error / max(|y|, 1 nM) < 1e-7)`, () => {
  for (const c of F.ode) {
    const r = simulate(c.stimulus, { duration: c.duration, dt: c.dt, intensity: c.intensity, params: c.params });
    assert.equal(r.t.length, c.n);
    for (let k = 0; k < 4; k++) for (let i = 0; i < c.n; i++) {
      const e = Math.abs(r.y[k][i] - c.y[k][i]) / Math.max(Math.abs(c.y[k][i]), 1.0);
      worstTight = Math.max(worstTight, e);
    }
  }
  assert.ok(worstTight < 1e-7, `worst relative error ${worstTight}`);
});

let worstRepo = 0;
ok('Ca_cyt matches the repo\'s simulate() (default odeint tolerances, 600 s) to < 1e-5 relative', () => {
  for (const c of F.repo_simulate) {
    const r = simulate(c.stimulus, { duration: 600, dt: 0.1 });
    c.Ca_cyt.forEach((v, j) => { worstRepo = Math.max(worstRepo, Math.abs(r.y[0][j * c.stride] - v) / Math.abs(v)); });
  }
  assert.ok(worstRepo < 1e-5, `worst relative difference ${worstRepo}`);
});

ok(`features, bandwidth, SNR, oscillations equal Python on ${F.signal.length} traces (1e-9)`, () => {
  for (const c of F.signal) {
    const t = c.ca.map((_, i) => i * c.dt);
    const f = extractFeatures(t, c.ca);
    close(Object.fromEntries(Object.entries(f).map(([k, v]) => [k, typeof v === 'number' ? nul(v) : v])), c.features, 1e-9, c.name);
    close(estimateBandwidth(t, c.ca), c.bandwidth, 1e-9, `${c.name} bandwidth`);
    close(nul(estimateSnr(c.ca)), c.snr, 1e-9, `${c.name} snr`);
    close(detectOscillations(t, c.ca), c.oscillation, 1e-9, `${c.name} oscillation`);
  }
});

ok(`entropy, binning and mutual information (plug-in and Miller-Madow) equal Python on ${F.info.length} data sets (1e-12)`, () => {
  for (const c of F.info) {
    if (c.kind === 'discrete') {
      close(entropyDiscrete(c.x), c.hx, 1e-12);
      close(mutualInformationDiscrete(c.x, c.y), c.mi, 1e-12);
      close(entropyMillerMadow(c.x), c.hx_mm, 1e-12);
      close(mutualInformationDiscrete(c.x, c.y, true), c.mi_mm, 1e-12);
    } else {
      assert.deepEqual(binUniform(c.x, c.n_bins), c.bins_x);
      close(mutualInformationContinuous(c.x, c.y, c.n_bins), c.mi, 1e-12);
      close(mutualInformationContinuous(c.x, c.y, c.n_bins, true), c.mi_mm, 1e-12);
    }
  }
});

for (const c of F.csv) {
  ok(`full trace analysis of "${c.name}" equals the Python API (1e-9)`, () => {
    close(analyseCsv(c.text, c.n_bins, c.units), c.result, 1e-9, c.name);
  });
}

for (const c of F.tables) {
  ok(`signature table "${c.name}": rows, features, summary, discrimination equal Python (1e-9)`, () => {
    close(analyseTable(parseSignatureCsv(c.text)), c.result, 1e-9, c.name);
  });
}

console.log(`\n${n} checks passed; ODE worst rel. error vs tight odeint ${worstTight.toExponential(2)}, `
  + `vs repo simulate() ${worstRepo.toExponential(2)}`);
