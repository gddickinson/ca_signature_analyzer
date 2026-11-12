// Static-site data layer: answers api() calls without the Python server.
// GET paths come from the JSON exported by scripts/build_static.py. POST /simulate, /analyze and /signatures run the
// JS port (cs_*.js, tested against the Python in cs_engine.test.mjs); the Python-only estimators of /signatures
// (k-NN mutual information, Random Forest) are reported as unavailable.
import { simulate as simJs } from './cs_model.js';
import { extractFeatures } from './cs_signal.js';
import { analyseCsv, parseSignatureCsv, analyseTable, clean } from './cs_analysis.js';

const files = new Map();
function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export function simulateLocal(b) {
  const r = simJs(b.stimulus || 'ABA', { duration: +(b.duration ?? 600), dt: +(b.dt ?? 0.1), intensity: +(b.intensity ?? 1), params: b.params || {} });
  const [Ca_cyt, Ca_ER, Ca_vac, IP3] = r.y;
  return { stimulus: b.stimulus, t: r.t, Ca_cyt, Ca_ER, Ca_vac, IP3, features: clean(extractFeatures(r.t, Ca_cyt)) };
}

export async function staticApi(path, body) {
  const clean_ = path.split('?')[0].replace(/^\/+/, '');
  if (clean_ === 'simulate') return simulateLocal(body || {});
  if (clean_ === 'analyze') return analyseCsv(body.text || '', body.n_bins ?? 10, body.units || 'nM');
  if (clean_ === 'signatures') {
    const res = analyseTable(parseSignatureCsv(body.text || ''));
    res.python = { unavailable: true };
    return res;
  }
  return file(clean_);
}
