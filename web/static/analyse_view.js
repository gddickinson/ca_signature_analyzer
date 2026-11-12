// Renders the result of api('/analyze') (web/analysis.py or cs_analysis.js): traces, metrics table, MI heatmap,
// CDPK activation, CSV export.
import { esc, download } from './core.js';
import { linePlot, heatmap, fmt, COLORS } from './plots.js';
import { FEATURES } from './feature_table.js';

const METRICS = [
  ...FEATURES.map(([k, label, unit]) => [r => r.features[k], label, unit]),
  [r => r.bandwidth_hz, 'Bandwidth (95% power)', 'Hz'],
  [r => r.snr_amplitude, 'SNR (amplitude)', ''],
  [r => r.capacity_bits_per_s, 'Shannon capacity W·log₂(1+SNR²)', 'bit/s'],
  [r => r.entropy_bits, 'Amplitude entropy (plug-in)', 'bit'],
  [r => r.entropy_bits_mm, 'Amplitude entropy (Miller–Madow)', 'bit'],
];

export function renderResults(host, res, { label, simulated }) {
  const names = res.columns.map(c => c.name), t = res.t;
  const info = res.info;
  host.innerHTML = `<h2>${simulated ? '<span class="sim-badge">SIMULATED</span> ' : ''}${esc(label)}</h2>
    <p class="small mute">${info.rows} rows used${info.dropped ? `, ${info.dropped} incomplete rows skipped` : ''}; time column
      "${esc(info.time_column)}", Δt = ${fmt(info.dt)} s; ${names.length} trace${names.length > 1 ? 's' : ''}; concentrations in nM${res.units === 'uM' ? ' (converted from μM)' : ''}; ${res.n_bins} bins.</p>
    ${(res.warnings || []).map(w => `<p class="static-note">${esc(w)}</p>`).join('')}
    <label><input type="checkbox" id="alog" checked> log scale</label><div id="tplot"></div>
    <h2>Metrics per trace</h2><div class="scroll"><table><thead><tr><th>Metric</th>${names.map(n => `<th class="num">${esc(n)}</th>`).join('')}</tr></thead>
    <tbody>${METRICS.map(([get, lab, unit]) => `<tr><th>${esc(lab)}${unit ? ` <span class="mute">(${unit})</span>` : ''}</th>${res.columns.map(c => `<td class="num">${fmt(get(c))}</td>`).join('')}</tr>`).join('')}
    </tbody></table></div>
    <p class="small mute">Entropy is out of a maximum of log₂(${res.n_bins}) = ${fmt(res.columns[0].max_entropy_bits)} bits. SNR is the
      repo's amplitude ratio; the capacity squares it to the power ratio Shannon–Hartley needs (see Format &amp; metrics).</p>
    <button id="mcsv" class="alt">Download metrics (CSV)</button>
    ${names.length > 1 ? `<h2>Mutual information between traces (bits)</h2>
      <p class="small mute">Uniform binning into ${res.n_bins} bins (InformationAnalyzer.mutual_information_continuous); the
        diagonal is each trace's binned entropy. Plug-in estimate and the Miller–Madow bias-corrected estimate (each entropy
        + (occupied bins − 1)/(2N ln 2)); time points are treated as independent samples, so both are optimistic for
        autocorrelated traces.</p>
      <div class="two"><div><h3>Plug-in</h3><div class="scroll">${heatmap(res.mi.matrix, res.mi.names, res.mi.names, { title: 'mutual information, plug-in' })}</div></div>
      <div><h3>Miller–Madow corrected</h3><div class="scroll">${heatmap(res.mi.matrix_mm, res.mi.names, res.mi.names, { title: 'mutual information, Miller-Madow' })}</div></div></div>` : ''}
    <h2>CDPK activation</h2>
    <p class="small mute">Default CDPK panel. Steady state (CDPKDecoder.steady_state_activation, [Ca]ⁿ / (K_dⁿ + [Ca]ⁿ)) at
      the trace's baseline and peak, and the dynamic activation along the trace (da/dt = k_off(([Ca]/K_d)ⁿ(1 − a) − a)):
      its peak and its time integral (s). Concentrations must be in nM for this to be meaningful.</p>
    <label for="csel">Trace</label><select id="csel" style="max-width:320px">${names.map((n, i) => `<option value="${i}">${esc(n)}</option>`).join('')}</select>
    <div class="scroll" id="cdpk"></div>`;
  const plot = () => {
    const log = host.querySelector('#alog').checked;
    host.querySelector('#tplot').innerHTML = linePlot(names.map((n, i) => ({ x: t, y: res.traces[n], label: n, color: COLORS[i % COLORS.length] })),
      { xlabel: 'time (s)', ylabel: '[Ca²⁺] (nM)', log, height: 280, legend: true });
  };
  const cdpk = () => {
    const c = res.columns[+host.querySelector('#csel').value];
    host.querySelector('#cdpk').innerHTML = `<table><thead><tr><th>CDPK</th><th class="num">K_d (nM)</th><th class="num">Hill n</th>
      <th class="num">at baseline (${fmt(c.features.baseline_ca)} nM)</th><th class="num">at peak (${fmt(c.features.peak_amplitude)} nM)</th>
      <th class="num">dynamic: peak</th><th class="num">dynamic: integral (s)</th></tr></thead>
      <tbody>${c.cdpk.map(d => `<tr><td>${esc(d.name)}</td><td class="num">${fmt(d.K_d)}</td><td class="num">${fmt(d.n_Hill)}</td>
      <td class="num">${fmt(d.at_baseline)}</td><td class="num">${fmt(d.at_peak)}</td>
      <td class="num">${fmt(d.peak_activation)}</td><td class="num">${fmt(d.integrated_activation)}</td></tr>`).join('')}</tbody></table>`;
  };
  host.querySelector('#alog').onchange = plot;
  host.querySelector('#csel').onchange = cdpk;
  host.querySelector('#mcsv').onclick = () => {
    const head = ['trace', ...METRICS.map(([, l, u]) => `${l}${u ? ` (${u})` : ''}`)];
    const q = s => `"${String(s).replace(/"/g, '""')}"`;
    const rows = res.columns.map(c => [q(c.name), ...METRICS.map(([g]) => { const v = g(c); return v === null || v === undefined ? '' : v; })].join(','));
    download('ca_signature_metrics.csv', [head.map(q).join(','), ...rows].join('\n') + '\n', 'text/csv');
  };
  plot(); cdpk();
}
