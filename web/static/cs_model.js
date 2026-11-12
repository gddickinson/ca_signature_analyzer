// CalciumDynamicsModel (src/biophysical_model.py) in JavaScript: the same stimulus patterns and ODE right-hand side
// (state [Ca_cyt, Ca_ER, Ca_vac, IP3], all nM, time in s). Python integrates with scipy's odeint (LSODA); here an
// adaptive Dormand-Prince 5(4) integrator with tight tolerances is used, stepping exactly to every output time.
// cs_engine.test.mjs checks derivatives() against Python to 1e-12 and trajectories against a tight-tolerance odeint.

// ModelParameters defaults (units in the Python source: nM/s, nM, 1/s, relative volumes)
export const DEFAULTS = {
  V_PM_GLR: 50.0, V_PM_CNGC: 30.0, V_ER_release: 100.0, V_vac_release: 80.0,
  V_PM_pump: 40.0, V_ER_pump: 60.0, V_vac_pump: 50.0,
  K_pump: 200.0, K_ER: 300.0, K_vac: 500.0,
  K_IP3: 100.0, n_IP3: 2.0, V_PLC: 50.0, k_IP3_deg: 0.5,
  B_total: 100000.0, K_buffer: 500.0, Ca_ER_max: 100000.0, Ca_vac_max: 500000.0,
  V_PM_leak: 8.0, k_ER_leak: 2.4e-4, k_vac_leak: 3.33e-5,
  vol_ratio_ER: 0.1, vol_ratio_vac: 10.0,
};
export const STIMULI = ['ABA', 'NaCl', 'mechanical_touch', 'flg22', 'cold_shock', 'H2O2', 'glutamate'];
export const Y0 = [100.0, 50000.0, 300000.0, 10.0];
export const STATE_NAMES = ['Ca_cyt', 'Ca_ER', 'Ca_vac', 'IP3'];

// CalciumDynamicsModel.stimulus_function
export function stimulus(t, type, intensity = 1.0) {
  const m = { GLR_activity: 0.0, CNGC_activity: 0.0, PLC_activity: 0.0 };
  if (type === 'ABA') {
    if (t > 60) { m.GLR_activity = intensity * 0.5; m.PLC_activity = intensity * 1.0; }
  } else if (type === 'NaCl') {
    if (t > 10) { m.GLR_activity = intensity * 0.8; m.CNGC_activity = intensity * 0.6; }
  } else if (type === 'mechanical_touch') {
    if (0.5 < t && t < 5) m.GLR_activity = intensity * 2.0;
  } else if (type === 'flg22') {
    if (t > 5) {
      if (t < 60) m.GLR_activity = intensity * 1.0;
      if (t > 30) m.PLC_activity = intensity * 0.8;
    }
  } else if (type === 'cold_shock') {
    if (2 < t && t < 60) m.GLR_activity = intensity * 1.2;
  } else if (type === 'H2O2') {
    if (t > 30) {
      m.PLC_activity = intensity * (0.5 + 0.5 * Math.sin(2 * Math.PI * 0.4 * (t - 30) / 60));
      m.GLR_activity = intensity * 0.3;
    }
  } else if (type === 'glutamate') {
    if (t > 1) m.GLR_activity = intensity * (1.0 - Math.exp(-(t - 1) / 5.0));
  }
  return m;
}

// CalciumDynamicsModel.derivatives (same flux terms and order)
export function derivatives(y, t, type, intensity, p = DEFAULTS, out = new Array(4)) {
  const [Ca_cyt, Ca_ER, Ca_vac, IP3] = y;
  const s = stimulus(t, type, intensity);
  const J_PM_in = p.V_PM_GLR * s.GLR_activity + p.V_PM_CNGC * s.CNGC_activity;
  const ipn = IP3 ** p.n_IP3;
  const J_ER_release = p.V_ER_release * (ipn / (p.K_IP3 ** p.n_IP3 + ipn)) * (Ca_ER / p.Ca_ER_max);
  const J_vac_release = p.V_vac_release * (Ca_vac / (p.K_vac + Ca_vac)) * s.GLR_activity;
  const c2 = Ca_cyt ** 2, k2 = p.K_pump ** 2;
  const J_PM_pump = p.V_PM_pump * (c2 / (k2 + c2));
  const J_ER_pump = p.V_ER_pump * (c2 / (k2 + c2));
  const J_vac_pump = p.V_vac_pump * (c2 / (k2 + c2));
  const J_PM_leak = p.V_PM_leak;
  const J_ER_leak = p.k_ER_leak * (Ca_ER - Ca_cyt);
  const J_vac_leak = p.k_vac_leak * (Ca_vac - Ca_cyt);
  const J_IP3_synth = p.V_PLC * s.PLC_activity;
  const J_IP3_deg = p.k_IP3_deg * IP3;
  // fluxes in cytosolic-concentration units; stores change by -J_net / vol_ratio (total Ca2+ conserved)
  const J_ER_net = J_ER_release + J_ER_leak - J_ER_pump;
  const J_vac_net = J_vac_release + J_vac_leak - J_vac_pump;
  out[0] = J_PM_in + J_PM_leak - J_PM_pump + J_ER_net + J_vac_net;
  out[1] = -J_ER_net / p.vol_ratio_ER;
  out[2] = -J_vac_net / p.vol_ratio_vac;
  out[3] = J_IP3_synth - J_IP3_deg;
  return out;
}

// Dormand-Prince 5(4) tableau
const C = [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1];
const A = [[], [1 / 5], [3 / 40, 9 / 40], [44 / 45, -56 / 15, 32 / 9],
  [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
  [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
  [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]];
const B = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0];
const E = [71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40]; // B - B*

// Integrate f from t0 to t1 starting at y (modified in place). h: suggested step (returned updated).
function dopriInterval(f, y, t0, t1, h, rtol, atol) {
  const n = y.length, K = Array.from({ length: 7 }, () => new Array(n)), ytmp = new Array(n), ynew = new Array(n);
  let t = t0, steps = 0;
  while (t < t1) {
    if (t + h > t1) h = t1 - t;
    f(y, t, K[0]);
    for (let s = 1; s < 7; s++) {
      for (let i = 0; i < n; i++) {
        let acc = y[i];
        for (let j = 0; j < s; j++) acc += h * A[s][j] * K[j][i];
        ytmp[i] = acc;
      }
      f(ytmp, t + C[s] * h, K[s]);
    }
    let err = 0;
    for (let i = 0; i < n; i++) {
      let acc = y[i], e = 0;
      for (let j = 0; j < 7; j++) { acc += h * B[j] * K[j][i]; e += h * E[j] * K[j][i]; }
      ynew[i] = acc;
      const sc = atol + rtol * Math.max(Math.abs(y[i]), Math.abs(acc));
      err = Math.max(err, Math.abs(e) / sc);
    }
    if (!Number.isFinite(err)) { // e.g. a trial stage with IP3 < 0 and a non-integer n_IP3: retry with a smaller step
      h *= 0.2;
      if (h < 1e-12 || ++steps > 1e6) throw new Error('The ODE solution became non-finite: check the parameters.');
      continue;
    }
    if (err <= 1) {
      t = (t + h >= t1 - 1e-12 * Math.max(1, Math.abs(t1))) ? t1 : t + h;
      for (let i = 0; i < n; i++) y[i] = ynew[i];
    }
    h *= Math.min(5, Math.max(0.2, 0.9 * Math.pow(err || 1e-10, -0.2)));
    if (++steps > 1e6) throw new Error('ODE integration did not converge.');
  }
  return h;
}

// CalciumDynamicsModel.simulate: t = arange(0, duration, dt); returns { t, y: [Ca_cyt[], Ca_ER[], Ca_vac[], IP3[]] }
export function simulate(type, { duration = 600.0, dt = 0.1, intensity = 1.0, params = {}, y0 = Y0,
  rtol = 1e-10, atol = 1e-8 } = {}) {
  if (!STIMULI.includes(type)) throw new Error(`Invalid stimulus_type '${type}'. Must be one of: ${STIMULI.join(', ')}`);
  if (!(dt > 0)) throw new Error(`dt must be > 0, got ${dt}`);
  if (intensity < 0) throw new Error(`intensity must be >= 0, got ${intensity}`);
  const p = { ...DEFAULTS, ...params };
  const n = Math.max(0, Math.ceil(duration / dt));
  const t = new Float64Array(n), out = STATE_NAMES.map(() => new Float64Array(n));
  const f = (yy, tt, d) => derivatives(yy, tt, type, intensity, p, d);
  const y = [...y0];
  let h = Math.min(dt, 0.01);
  for (let i = 0; i < n; i++) {
    t[i] = i * dt;
    if (i > 0) h = dopriInterval(f, y, t[i - 1], t[i], h, rtol, atol);
    for (let k = 0; k < 4; k++) out[k][i] = y[k];
  }
  return { t, y: out };
}
