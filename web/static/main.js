// Entry point: registers lazily loaded pages, sets the footer mode text and starts the router.
import { start, lazy, route, app, isStatic, $ } from './core.js';
import { startBackground } from './background.js';

const PAGES = { 'home.js': ['home'], 'model.js': ['model'], 'analyse.js': ['analyse'],
  'signatures.js': ['signatures'], 'metrics.js': ['metrics'], 'about.js': ['about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: model and analyses run in your browser (JS port tested against the Python)'
  : 'local app: analyses run in Python';
startBackground();
start();
