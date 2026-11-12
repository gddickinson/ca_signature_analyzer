// Model explorer: CalciumDynamicsModel (src/biophysical_model.py) run live in the browser (cs_model.js), with every
// parameter that enters the equations on a slider, the four state traces and the signature features.
// On the local app, "Compare with Python" runs the same settings through the server (scipy odeint).
import { route, app, api, esc, isStatic, slider, download } from './core.js';
import { simulateLocal } from './static_api.js';
import { linePlot, fmt } from './plots.js';
import { featureRows } from './feature_table.js';

const state = { stimulus: 'ABA', intensity: 1.0, duration: 600, params: {}, log: false };

function range(p) {
  if (p.name === 'n_IP3') return { min: 0.5, max: 4, step: 0.1 };
  const floor = /^(K_|Ca_ER_max|vol_ratio)/.test(p.name) ? p.default / 20 : 0;
  return { min: floor, max: p.default * 3, step: p.default / 100 };
}

route('model', async () => {
  const M = await api('/model');
  const used = M.parameters.filter(p => p.used), unused = M.parameters.filter(p => !p.used);
  const groups = [...new Set(used.map(p => p.group))];
  app().innerHTML = `<h1 class="glow">Model explorer</h1>
    <p class="lead">The repo's ODE model of plant Ca²⁺ dynamics: plasma-membrane GLR and CNGC influx, InsP3-induced ER
      release, vacuolar release, three Ca-ATPases and InsP3 synthesis by PLC. State: [Ca_cyt, Ca_ER, Ca_vac, IP3] in nM.</p>
    <div class="side"><div class="panel">
      <label for="stim">Stimulus</label><select id="stim">${M.stimuli.map(s => `<option ${s.name === state.stimulus ? 'selected' : ''}>${esc(s.name)}</option>`).join('')}</select>
      <div id="sl-main"></div>
      <label><input type="checkbox" id="logy" ${state.log ? 'checked' : ''}> log scale for cytosolic Ca²⁺</label>
      <button id="reset" class="alt">Reset parameters</button>
      <div class="params" id="params">${groups.map(g => `<h3>${esc(g)}</h3><div data-group="${esc(g)}"></div>`).join('')}</div>
      <p class="small mute">Defined in ModelParameters but not used by the equations (no effect):
        ${unused.map(p => `<code>${esc(p.name)}</code>`).join(', ')}.</p>
    </div><div>
      <p><span class="sim-badge">SIMULATED</span> <span id="desc" class="small"></span></p>
      <div id="plots"></div>
      <h2>Signature features</h2><div class="scroll" id="feats"></div>
      <button id="csv" class="alt">Download traces (CSV)</button>${isStatic() ? '' : '<button id="py" class="alt">Compare with Python</button>'}
      <p id="cmp" class="small mute"></p>
    </div></div>`;
  const main = document.getElementById('sl-main');
  slider(main, { id: 'int', label: 'Intensity', min: 0, max: 3, step: 0.05, value: state.intensity, fmt: v => v.toFixed(2) }, v => { state.intensity = v; schedule(); });
  slider(main, { id: 'dur', label: 'Duration', min: 60, max: 1800, step: 30, value: state.duration, unit: ' s' }, v => { state.duration = v; schedule(); });
  const inputs = {};
  for (const p of used) {
    const host = document.querySelector(`[data-group="${CSS.escape(p.group)}"]`);
    const val = state.params[p.name] ?? p.default;
    inputs[p.name] = slider(host, { id: p.name, label: `${p.name} (${p.unit || 'dimensionless'}) ${p.comment ? '· ' + p.comment : ''}`, ...range(p), value: val, fmt: v => fmt(v) },
      v => { state.params[p.name] = v; schedule(); });
  }
  document.getElementById('stim').onchange = e => { state.stimulus = e.target.value; schedule(); };
  document.getElementById('logy').onchange = e => { state.log = e.target.checked; schedule(); };
  document.getElementById('reset').onclick = () => {
    state.params = {};
    for (const p of used) { inputs[p.name].value = p.default; inputs[p.name].dispatchEvent(new Event('input')); }
  };
  let last = null, pending = false;
  function schedule() { if (!pending) { pending = true; requestAnimationFrame(() => { pending = false; run(); }); } }
  function run() {
    const desc = M.stimuli.find(s => s.name === state.stimulus).description;
    document.getElementById('desc').textContent = `${state.stimulus}, intensity ${state.intensity.toFixed(2)}: ${desc}`;
    try {
      last = simulateLocal({ stimulus: state.stimulus, duration: state.duration, dt: 0.1, intensity: state.intensity, params: state.params });
    } catch (e) { document.getElementById('plots').innerHTML = `<p class="err">${esc(e.message)}</p>`; return; }
    const f = last.features, t = last.t;
    const refs = f.baseline_ca !== null ? [{ y: f.baseline_ca, label: 'baseline' },
      { y: f.baseline_ca + 0.5 * (f.peak_amplitude - f.baseline_ca), label: 'half-max' }] : [];
    document.getElementById('plots').innerHTML = linePlot([{ x: t, y: last.Ca_cyt, label: 'Ca_cyt' }],
      { title: 'Cytosolic Ca²⁺', xlabel: 'time (s)', ylabel: '[Ca²⁺]cyt (nM)', log: state.log, refs, height: 260 })
      + `<div class="panels">${[['Ca_ER', 'ER Ca²⁺', '#ffcf3f'], ['Ca_vac', 'Vacuolar Ca²⁺', '#4fd1ff'], ['IP3', 'InsP3', '#ff6fae']].map(([k, ttl, c]) =>
        `<div>${linePlot([{ x: t, y: last[k].map(v => (k === 'IP3' ? v : v / 1000)), label: k, color: c }],
          { title: ttl, xlabel: 'time (s)', ylabel: k === 'IP3' ? 'nM' : 'μM', height: 210, width: 400 })}</div>`).join('')}</div>`;
    document.getElementById('feats').innerHTML = `<table><tbody>${featureRows(f)}</tbody></table>`;
  }
  document.getElementById('csv').onclick = () => {
    if (!last) return;
    const lines = ['time_s,Ca_cyt_nM,Ca_ER_nM,Ca_vac_nM,IP3_nM'];
    for (let i = 0; i < last.t.length; i++) lines.push([last.t[i], last.Ca_cyt[i], last.Ca_ER[i], last.Ca_vac[i], last.IP3[i]].join(','));
    download(`simulated_${state.stimulus}_I${state.intensity}.csv`, `# SIMULATED: CalciumDynamicsModel, ${state.stimulus}, intensity ${state.intensity}\n` + lines.join('\n') + '\n', 'text/csv');
  };
  const py = document.getElementById('py');
  if (py) py.onclick = async () => {
    const cmp = document.getElementById('cmp');
    cmp.textContent = 'Running in Python...';
    try {
      const r = await api('/simulate', { body: { stimulus: state.stimulus, duration: state.duration, dt: 0.1, intensity: state.intensity, params: state.params } });
      let worst = 0;
      r.Ca_cyt.forEach((v, i) => { worst = Math.max(worst, Math.abs(v - last.Ca_cyt[i]) / Math.max(Math.abs(v), 1)); });
      cmp.textContent = `Python (scipy odeint) vs this page: largest relative difference in Ca_cyt ${worst.toExponential(2)}; Python peak ${fmt(r.features.peak_amplitude)} nM, latency ${fmt(r.features.latency)} s.`;
    } catch (e) { cmp.textContent = e.message; }
  };
  run();
});
