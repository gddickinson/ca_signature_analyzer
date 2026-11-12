// Home page: hero, counts from the repo, and links to the pages.
import { route, app, api, esc, isStatic } from './core.js';

route('home', async () => {
  const info = await api('/info');
  app().innerHTML = `<section class="hero"><p class="kicker">Plant Ca²⁺ signalling · information theory</p>
    <h1 class="glow">${esc(info.title)}</h1>
    <p class="lead">${esc(info.description)} Explore the biophysical model with live traces, measure signature
      features and information-theoretic quantities on your own [Ca²⁺] traces, and compare signature tables.</p>
    <a class="btn" href="#/model">Explore the model</a><a class="btn alt" href="#/analyse">Analyse your traces</a></section>
    <div class="tiles">
      <div class="tile"><b>${info.n_stimuli}</b><span>stimulus types in the model</span></div>
      <div class="tile"><b>${info.n_state}</b><span>state variables (cytosol, ER, vacuole, InsP3)</span></div>
      <div class="tile"><b>${info.n_parameters}</b><span>model parameters</span></div>
      <div class="tile"><b>${info.n_cdpks}</b><span>CDPK decoders in the default panel</span></div>
      <div class="tile"><b>${info.n_examples}</b><span>literature-derived example signatures</span></div>
      <div class="tile"><b>${isStatic() ? 'JS' : 'Python'}</b><span>engine on this ${isStatic() ? 'site' : 'app'}</span></div></div>
    <div class="grid">
      <a class="card" href="#/model"><b>Model explorer</b><span>CalciumDynamicsModel: pick a stimulus, change intensity and any parameter, and watch cytosolic, ER and vacuolar Ca²⁺ and InsP3 update with the signature features.</span></a>
      <a class="card" href="#/analyse"><b>Analyse traces</b><span>Upload a CSV (time + one or more [Ca²⁺] columns) or use simulated demo traces: signature features, bandwidth, SNR, Shannon capacity, entropy, mutual information, CDPK activation.</span></a>
      <a class="card" href="#/signatures"><b>Signature tables</b><span>The example database and a simulated signature set: summary statistics, pairwise discrimination, mutual information per feature, classification accuracy.</span></a>
      <a class="card" href="#/metrics"><b>Format &amp; metrics</b><span>The input formats and what each metric means, as computed by the repo's code.</span></a>
      <a class="card" href="#/about"><b>About</b><span>What the toolkit is, how to run the desktop GUI, the scripts and this web app.</span></a>
    </div>`;
});
