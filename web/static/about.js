// About: what the toolkit is, how to run the desktop GUI, the scripts and this web app; credits from the README.
import { route, app, isStatic } from './core.js';

route('about', () => {
  app().innerHTML = `<h1 class="glow">About</h1>
  <p class="lead">Ca²⁺ Signature Analyzer is a Python toolkit for analyzing plant calcium signaling dynamics using
    information theory and biophysical modeling: a signature database, an ODE model of Ca²⁺ dynamics with seven stimulus
    types, CDPK decoder models, information-theoretic measures and plots, with a desktop GUI and command-line scripts.</p>

  <h2>This web interface</h2>
  <p>${isStatic()
    ? 'You are on the static site (GitHub Pages). The model and every analysis except the scikit-learn k-NN mutual information and the Random-Forest classifier run in your browser, in a JavaScript port of the repo\'s Python that is tested against it (node test against values written by the Python, agreement to 1e-9 or better for the analyses; ODE trajectories agree with scipy\'s odeint to about 1e-6). Uploaded files never leave your computer. The two Python-only estimators are shown precomputed for the built-in tables.'
    : 'You are on the local app: analyses of uploaded files run in the repo\'s Python code on this computer; the model explorer runs the JavaScript port for live updates and can compare itself with the Python.'}</p>
  <pre>pip install -r requirements.txt -r requirements-web.txt
python run_web.py            # http://127.0.0.1:8134
python scripts/build_static.py   # static site in site/</pre>

  <h2>Desktop GUI</h2>
  <pre>python src/gui_app.py        # or: python launch_gui.py</pre>
  <p>Tabs: Database, Simulate Signatures, Decoder Analysis, Information Theory, Visualizations.</p>

  <h2>Command-line scripts</h2>
  <pre>python run_simulation.py --stimulus ABA --duration 600 --output outputs/simulation.csv
python analyze_database.py --input data/signatures.csv --output outputs/analysis</pre>

  <h2>Source and licence</h2>
  <p>Code on <a href="https://github.com/gddickinson/ca_signature_analyzer">GitHub</a>, MIT licence. The example signatures
    cite their sources in the table (Allen et al. 2001; Knight et al. 1996, 1997; Legue et al. 1997; Ranf et al. 2011;
    Pei et al. 2000; Toyota et al. 2018). Developed as part of research on plant calcium signaling and information theory
    analysis; built with AI assistance from Claude (Anthropic).</p>`;
});
