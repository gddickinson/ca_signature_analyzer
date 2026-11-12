// Small SVG chart helpers (no libraries): line plots with optional log y-axis and reference lines, and heatmaps.
import { esc } from './core.js';

export const COLORS = ['#7dff6a', '#ffcf3f', '#4fd1ff', '#ff6fae', '#b18cff', '#ff9a3c', '#3dffd0', '#e8f1ff'];

export function fmt(v, d = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  if (typeof v === 'boolean') return v ? 'yes' : 'no';
  if (typeof v !== 'number') return esc(v);
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '−∞';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e6 || a < 1e-3)) return v.toExponential(2);
  return String(+v.toPrecision(Math.min(Math.max(d + 1, 2), 8)));
}

function ticks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9 * span; v += step) out.push(+v.toPrecision(12));
  return out;
}

// Keep min and max of each pixel bucket so narrow spikes survive downsampling.
function decimate(x, y, buckets) {
  if (x.length <= buckets * 2) return [...x.keys()];
  const idx = [], per = x.length / buckets;
  for (let b = 0; b < buckets; b++) {
    const s = Math.floor(b * per), e = Math.min(x.length, Math.floor((b + 1) * per));
    let mi = s, ma = s;
    for (let i = s; i < e; i++) { if (y[i] < y[mi]) mi = i; if (y[i] > y[ma]) ma = i; }
    idx.push(Math.min(mi, ma)); if (mi !== ma) idx.push(Math.max(mi, ma));
  }
  return idx;
}

// series: [{ x, y, label, color }]; opts: { title, xlabel, ylabel, log, refs: [{ y, label }], height }
export function linePlot(series, opts = {}) {
  const W = opts.width || 720, H = opts.height || 240, L = 64, R = 14, T = opts.title ? 24 : 10, B = 40;
  const tf = v => (opts.log ? Math.log10(Math.max(v, 1e-6)) : v);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of series) for (let i = 0; i < s.x.length; i++) {
    const yv = tf(s.y[i]);
    if (!Number.isFinite(yv)) continue;
    x0 = Math.min(x0, s.x[i]); x1 = Math.max(x1, s.x[i]); y0 = Math.min(y0, yv); y1 = Math.max(y1, yv);
  }
  for (const r of opts.refs || []) { y0 = Math.min(y0, tf(r.y)); y1 = Math.max(y1, tf(r.y)); }
  if (!Number.isFinite(x0)) return '<p class="mute">No data to plot.</p>';
  if (y1 - y0 < 1e-12) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.05; y0 -= pad; y1 += pad;
  const X = v => L + (v - x0) / (x1 - x0 || 1) * (W - L - R), Y = v => T + (1 - (tf(v) - y0) / (y1 - y0)) * (H - T - B);
  let g = '';
  for (const v of ticks(x0, x1)) g += `<line class="grid-l" x1="${X(v)}" x2="${X(v)}" y1="${T}" y2="${H - B}"/><text x="${X(v)}" y="${H - B + 14}" text-anchor="middle">${fmt(v)}</text>`;
  const yt = opts.log ? ticks(Math.ceil(y0), Math.floor(y1)).filter(Number.isInteger) : ticks(y0, y1);
  for (const v of yt) {
    const yy = T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
    g += `<line class="grid-l" x1="${L}" x2="${W - R}" y1="${yy}" y2="${yy}"/><text x="${L - 6}" y="${yy + 4}" text-anchor="end">${opts.log ? fmt(10 ** v) : fmt(v)}</text>`;
  }
  for (const r of opts.refs || []) g += `<line class="ref-l" x1="${L}" x2="${W - R}" y1="${Y(r.y)}" y2="${Y(r.y)}"/><text x="${W - R - 4}" y="${Y(r.y) - 4}" text-anchor="end">${esc(r.label)}</text>`;
  const lines = series.map((s, k) => {
    const idx = decimate(s.x, s.y, 700);
    const d = idx.map((i, j) => `${j ? 'L' : 'M'}${X(s.x[i]).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join('');
    return `<path d="${d}" fill="none" stroke="${s.color || COLORS[k % COLORS.length]}" stroke-width="1.6"/>`;
  }).join('');
  const legend = series.length > 1 || opts.legend ? `<div class="legend">${series.map((s, k) => `<span><i style="background:${s.color || COLORS[k % COLORS.length]}"></i>${esc(s.label)}</span>`).join('')}</div>` : '';
  return `<svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.title || opts.ylabel || 'plot')}">
    ${opts.title ? `<text class="axis-label" x="${L}" y="16">${esc(opts.title)}</text>` : ''}
    <rect class="axis-box" x="${L}" y="${T}" width="${W - L - R}" height="${H - T - B}"/>${g}${lines}
    <text class="axis-label" x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">${esc(opts.xlabel || '')}</text>
    <text class="axis-label" transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${esc(opts.ylabel || '')}</text></svg>${legend}`;
}

// matrix heatmap with values in the cells; opts: { title, max, digits }
export function heatmap(matrix, rows, cols = rows, opts = {}) {
  const n = rows.length, m = cols.length, cell = Math.max(26, Math.min(64, 520 / Math.max(n, m)));
  const L = 120, T = 96, W = L + m * cell + 10, H = T + n * cell + 10;
  let max = opts.max ?? 0;
  if (!opts.max) matrix.forEach(r => r.forEach(v => { if (Number.isFinite(v)) max = Math.max(max, v); }));
  const color = v => { const a = Math.max(0, Math.min(1, (v ?? 0) / (max || 1))); return `rgba(125,255,106,${(0.08 + 0.85 * a).toFixed(3)})`; };
  let s = '';
  matrix.forEach((r, i) => r.forEach((v, j) => {
    s += `<rect x="${L + j * cell}" y="${T + i * cell}" width="${cell - 2}" height="${cell - 2}" fill="${color(v)}"><title>${esc(rows[i])} / ${esc(cols[j])}: ${fmt(v)}</title></rect>`;
    if (cell >= 30) s += `<text class="heat-t" x="${L + j * cell + cell / 2 - 1}" y="${T + i * cell + cell / 2 + 3}" text-anchor="middle">${fmt(v, opts.digits ?? 2)}</text>`;
  }));
  rows.forEach((r, i) => { s += `<text x="${L - 6}" y="${T + i * cell + cell / 2 + 3}" text-anchor="end">${esc(String(r).slice(0, 16))}</text>`; });
  cols.forEach((c, j) => { s += `<text transform="translate(${L + j * cell + cell / 2} ${T - 6}) rotate(-50)">${esc(String(c).slice(0, 16))}</text>`; });
  return `<svg class="plot" viewBox="0 0 ${W} ${H}" style="max-width:${W}px" role="img" aria-label="${esc(opts.title || 'heatmap')}">${s}</svg>`;
}

// Horizontal bar chart: items [{ label, value }]
export function bars(items, opts = {}) {
  const W = 640, row = 22, L = 150, H = items.length * row + 10;
  const max = opts.max ?? Math.max(...items.map(d => d.value || 0), 1e-12);
  const s = items.map((d, i) => `<text x="${L - 6}" y="${i * row + 16}" text-anchor="end">${esc(d.label)}</text>
    <rect x="${L}" y="${i * row + 4}" height="${row - 8}" width="${Math.max(0, (d.value || 0) / max * (W - L - 70))}" fill="${opts.color || '#4fd1ff'}"/>
    <text x="${L + Math.max(0, (d.value || 0) / max * (W - L - 70)) + 6}" y="${i * row + 16}">${fmt(d.value)}</text>`).join('');
  return `<svg class="plot" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.title || 'bar chart')}">${s}</svg>`;
}
