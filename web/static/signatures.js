// Signature tables (SignatureDatabase): the repo's example database, a SIMULATED table from the model, or the user's
// CSV. Summary statistics, feature matrix and pairwise discrimination run in JS on the static site; the k-NN mutual
// information (estimate_signature_capacity) and Random-Forest classification are Python only (precomputed for the
// two built-in tables).
import { route, app, api, esc, isStatic, needsServer, go } from './core.js';
import { heatmap, bars, fmt } from './plots.js';

const SOURCES = {
  example: ['Example database', 'create_example_database(): seven literature-derived signatures, one per stimulus (data/example_signatures.csv).'],
  simulated: ['Simulated set', 'CalciumDynamicsModel, 7 stimuli × intensities 0.5, 0.75, 1, 1.25, 1.5 (600 s, dt 0.1 s), features from extract_signature_features, stored as the GUI\'s "Save simulation to database" does (missing rise/decay/latency → 0).'],
  upload: ['Your table', 'A CSV in the SignatureDatabase format (see Format & metrics).'],
};

route('signatures', async (args) => {
  const src = SOURCES[args[0]] ? args[0] : 'example';
  app().innerHTML = `<h1 class="glow">Signature tables</h1>
    <p class="lead">Tables of signature features (one row per recorded or simulated response) and how well the stimuli
      can be told apart from them.</p>
    <div class="seg" role="tablist">${Object.entries(SOURCES).map(([k, [l]]) => `<button role="tab" aria-selected="${k === src}" class="${k === src ? 'on' : ''}" data-src="${k}">${esc(l)}</button>`).join('')}</div>
    <p class="small">${src === 'simulated' ? '<span class="sim-badge">SIMULATED</span> ' : ''}${esc(SOURCES[src][1])}</p>
    ${src === 'upload' ? `<div class="panel"><label for="sfile">Signature CSV</label><input type="file" id="sfile" accept=".csv,text/csv">
      <button id="sgo">Analyse table</button><p id="smsg" class="small mute"></p></div>` : ''}
    <div id="sout"></div>`;
  document.querySelectorAll('[data-src]').forEach(b => { b.onclick = () => go('#/signatures/' + b.dataset.src); });
  const out = document.getElementById('sout');
  if (src === 'example') render(out, await api('/database'));
  else if (src === 'simulated') render(out, await api('/demo_signatures'));
  else document.getElementById('sgo').onclick = async () => {
    const f = document.getElementById('sfile').files[0], msg = document.getElementById('smsg');
    if (!f) { msg.innerHTML = '<span class="err">Choose a CSV file first.</span>'; return; }
    msg.textContent = 'Analysing...';
    try { render(out, await api('/signatures', { body: { text: await f.text() } })); msg.textContent = ''; }
    catch (e) { msg.innerHTML = `<span class="err">${esc(e.message)}</span>`; }
  };
});

const COLS = [['signature_id', 'ID'], ['stimulus', 'Stimulus'], ['cell_type', 'Cell type'], ['baseline_ca', 'Baseline (nM)'],
  ['peak_amplitude', 'Peak (nM)'], ['fold_change', 'Fold'], ['rise_time', 'Rise (s)'], ['decay_tau', 'Decay τ (s)'],
  ['duration', 'Duration (s)'], ['latency', 'Latency (s)'], ['oscillatory', 'Osc.'], ['frequency', 'Freq. (Hz)'],
  ['reference', 'Reference']];

function render(host, d) {
  const s = d.summary;
  const one = d.class_counts.every(c => c < 2);
  host.innerHTML = `<div class="tiles">
      <div class="tile"><b>${s.total_signatures}</b><span>signatures</span></div>
      <div class="tile"><b>${s.unique_stimuli}</b><span>stimuli</span></div>
      <div class="tile"><b>${s.unique_cell_types}</b><span>cell types</span></div>
      <div class="tile"><b>${fmt(s.oscillatory_fraction * 100, 2)}%</b><span>oscillatory</span></div>
      <div class="tile"><b>${fmt(s.mean_peak_amplitude, 2)}</b><span>mean peak (nM)</span></div>
      <div class="tile"><b>${fmt(s.mean_duration, 2)}</b><span>mean duration (s)</span></div></div>
    <div class="scroll"><table><thead><tr>${COLS.map(([, l]) => `<th>${esc(l)}</th>`).join('')}</tr></thead><tbody>
      ${d.rows.map(r => `<tr>${COLS.map(([k]) => `<td class="${typeof r[k] === 'number' ? 'num' : 'small'}">${fmt(r[k])}</td>`).join('')}</tr>`).join('')}
    </tbody></table></div>
    <h2>Pairwise discrimination</h2>
    <p class="small mute">SignatureDiscriminability.pairwise_discrimination on the 9-feature vectors: the distance between
      two stimuli's mean feature vectors, each feature divided by the mean of the two classes' standard deviations.
      ${one ? '<span class="err">Every stimulus has a single signature here, so the standard deviations are 0 (floored at 1e-10) and the distances are meaningless (~1e10).</span>' : ''}</p>
    <div class="scroll">${heatmap(d.discrimination, d.classes, d.classes, { title: 'discrimination' })}</div>
    <h2>Information and classification (Python)</h2><div id="py"></div>`;
  const py = host.querySelector('#py'), p = d.python || {};
  if (p.unavailable) { py.innerHTML = needsServer('The k-NN mutual-information estimate and the Random-Forest classifier'); return; }
  let h = '';
  if (p.capacity) {
    const c = p.capacity;
    h += `<p>InformationAnalyzer.estimate_signature_capacity: mutual information between each feature and the stimulus
      (scikit-learn's k-nearest-neighbour estimator, mutual_info_classif), in bits.</p>
      <div class="tiles"><div class="tile"><b>${fmt(c.max_information_bits)}</b><span>max = log₂(${c.n_stimuli} stimuli) bits</span></div>
      <div class="tile"><b>${fmt(c.avg_mutual_information_bits)}</b><span>mean MI per feature (bits)</span></div>
      <div class="tile"><b>${fmt(c.information_efficiency * 100, 2)}%</b><span>efficiency = mean MI / max</span></div></div>
      ${bars(d.feature_names.map((f, i) => ({ label: f, value: c.mi_per_feature[i] })), { title: 'MI per feature (bits)' })}`;
  } else h += `<p class="small mute">estimate_signature_capacity could not run on this table (Python: ${esc(p.capacity_error)}).</p>`;
  if (p.classification) {
    const r = p.classification;
    h += `<p>SignatureDiscriminability.classification_accuracy, Random Forest (100 trees), 5-fold cross-validation:
      accuracy <b>${fmt(r.accuracy_mean)} ± ${fmt(r.accuracy_std)}</b> (SD over folds; ${r.n_samples} signatures, ${r.n_classes} classes).
      Information transmitted = mutual information between true and predicted stimulus (the confusion matrix):
      <b>${fmt(r.information_bits)}</b> bits plug-in, <b>${fmt(r.information_bits_miller_madow)}</b> bits Miller–Madow corrected,
      of at most ${fmt(r.max_information_bits)} bits.</p>
      <div class="scroll">${heatmap(r.confusion_matrix, d.classes, d.classes, { title: 'confusion matrix', digits: 0 })}</div>
      <p class="small mute">Confusion matrix: rows true stimulus, columns predicted.</p>`;
  } else h += `<p class="small mute">classification_accuracy could not run on this table (Python: ${esc(p.classification_error)}).</p>`;
  py.innerHTML = h;
  if (isStatic() && !d.python) py.innerHTML = needsServer('This analysis');
}
