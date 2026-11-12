// Analyse traces: the user's CSV (time + [Ca2+] columns) or SIMULATED demo traces from the repo's model, sent to
// api('/analyze'): Python on the local app, the JS port (cs_analysis.js) on the static site. Results: analyse_view.js.
import { route, app, api, esc, download, isStatic } from './core.js';
import { simulate, STIMULI } from './cs_model.js';
import { renderResults } from './analyse_view.js';

// Demo CSV: every stimulus, CalciumDynamicsModel defaults, intensity 1, 600 s, dt 0.1 s (deterministic, no noise).
export function demoCsv() {
  const runs = STIMULI.map(s => simulate(s, { duration: 600, dt: 0.1, intensity: 1.0 }));
  const lines = ['# SIMULATED demo traces: CalciumDynamicsModel (ca_signature_analyzer), default parameters, intensity 1,',
    '# deterministic ODE without noise. Columns: time (s), cytosolic [Ca2+] (nM) per stimulus.',
    'time_s,' + STIMULI.join(',')];
  const t = runs[0].t;
  for (let i = 0; i < t.length; i++) lines.push([+t[i].toFixed(10), ...runs.map(r => r.y[0][i])].join(','));
  return lines.join('\n') + '\n';
}

route('analyse', async (args) => {
  app().innerHTML = `<h1 class="glow">Analyse traces</h1>
    <p class="lead">Signature features, bandwidth, signal-to-noise ratio, Shannon capacity, entropy and mutual information
      for each [Ca²⁺] trace in a CSV file, plus steady-state CDPK activation at baseline and peak. See
      <a href="#/metrics">Format &amp; metrics</a> for the file format and the definitions.</p>
    <div class="panel"><div class="row">
      <div><label for="file">CSV file: time (s) in the first column, one [Ca²⁺] trace per further column</label>
        <input type="file" id="file" accept=".csv,.tsv,.txt,text/csv"></div>
      <div style="max-width:160px"><label for="units">Concentration unit</label>
        <select id="units"><option value="nM">nM</option><option value="uM">μM (× 1000)</option></select></div>
      <div style="max-width:160px"><label for="bins">Bins (entropy, MI)</label>
        <input type="number" id="bins" min="2" max="100" value="10"></div></div>
      <button id="go">Analyse file</button><button id="demo" class="alt">Use simulated demo traces</button>
      <button id="dl" class="alt">Download demo CSV</button>
      <p id="msg" class="small mute">${isStatic() ? 'Files are analysed in your browser and not uploaded anywhere.' : 'Files go only to the local Python server on this computer.'}</p></div>
    <div id="out"></div>`;
  const msg = document.getElementById('msg'), out = document.getElementById('out');
  const params = () => ({ n_bins: +document.getElementById('bins').value, units: document.getElementById('units').value });
  async function run(text, label, simulated) {
    msg.textContent = `Analysing ${label}...`; out.innerHTML = '';
    try {
      const res = await api('/analyze', { body: { text, ...params() } });
      msg.textContent = '';
      renderResults(out, res, { label, simulated });
    } catch (e) { msg.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  }
  document.getElementById('go').onclick = async () => {
    const f = document.getElementById('file').files[0];
    if (!f) { msg.innerHTML = '<span class="err">Choose a CSV file first.</span>'; return; }
    run(await f.text(), f.name, false);
  };
  document.getElementById('demo').onclick = () => {
    document.getElementById('units').value = 'nM';
    run(demoCsv(), 'simulated demo traces (7 stimuli)', true);
  };
  document.getElementById('dl').onclick = () => download('simulated_demo_traces.csv', demoCsv(), 'text/csv');
  if (args[0] === 'demo') document.getElementById('demo').click();
});
