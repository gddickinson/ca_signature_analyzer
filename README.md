# Ca²⁺ Signature Analyzer

A Python toolkit for analyzing plant calcium signaling dynamics using information theory and biophysical modeling.

## Features

- **Signature Database Management**: Store, load, and analyze Ca²⁺ signatures
- **Biophysical Modeling**: ODE-based simulation of Ca²⁺ dynamics with multiple stimuli
- **Decoder Analysis**: Model CDPK and other Ca²⁺ decoder responses
- **Information Theory**: Calculate mutual information, channel capacity, and discriminability
- **Comprehensive Visualization**: UMAP projections, heatmaps, signature comparisons
- **Dual Interface**: Both GUI and command-line scripts available

## Installation

### Requirements

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone or download this repository
2. Navigate to the project directory:
```bash
cd ca_signature_analyzer
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- umap-learn >= 0.5.0

## Usage

### GUI Application

Launch the graphical interface:

```bash
python src/gui_app.py
```

The GUI provides five main tabs:

1. **Database**: Load, view, and manage signature databases
2. **Simulate Signatures**: Run biophysical model simulations
3. **Decoder Analysis**: Analyze CDPK responses to signatures
4. **Information Theory**: Calculate MI and classification accuracy
5. **Visualizations**: Generate UMAP, heatmaps, and reports

### Command-Line Scripts

#### Run a Simulation

```bash
python run_simulation.py --stimulus ABA --duration 600 --output outputs/simulation.csv
```

Options:
- `--stimulus`: Stimulus type (ABA, NaCl, mechanical_touch, flg22, cold_shock, H2O2, glutamate)
- `--duration`: Simulation duration in seconds (default: 600)
- `--intensity`: Stimulus intensity (default: 1.0)
- `--output`: Output CSV file path
- `--plot`: Output plot file path

#### Analyze a Database

```bash
python analyze_database.py --input data/signatures.csv --output outputs/analysis
```

Options:
- `--input`: Input CSV file (if not provided, uses example database)
- `--output`: Output directory for results

This performs:
- Database statistics
- Signature simulation
- Decoder analysis
- Information theory calculations
- UMAP projection
- Classification testing
- Information flow analysis

### Python API

You can also use the modules directly in Python:

```python
from src.signature_database import SignatureDatabase, create_example_database
from src.biophysical_model import CalciumDynamicsModel
from src.decoder_model import DecoderPanel
from src.information_theory import InformationAnalyzer

# Load example database
db = create_example_database()
print(f"Loaded {len(db)} signatures")

# Run simulation
model = CalciumDynamicsModel()
t, sol = model.simulate('ABA', duration=600)
ca_cyt = sol[:, 0]

# Analyze with decoders
panel = DecoderPanel()
responses = panel.analyze_signature(t, ca_cyt)

# Calculate mutual information
features = db.get_feature_matrix()
labels = [sig.stimulus for sig in db.signatures]
# ... continue analysis
```

## Changes to results (2026-11-11)

Four fixes change numbers by design (details in SESSION_LOG.md):
- the ODE model now conserves Ca²⁺ between cytosol, ER and vacuole (fluxes in cytosolic units, stores change by
  J / volume ratio) and has passive leaks, so a resting state exists near 100 nM; ABA no longer runs away to 72 μM;
- CDPK dynamic activation follows the documented Hill curve (K_d, n) with k_off setting the speed, integrated stably
  (it used to flip 0/1 every step);
- example oscillation frequencies are stored in Hz (they were per-minute values labelled Hz);
- information flow bins the initial entropy like the MI, classification "information" is the MI of the confusion
  matrix, and a Miller–Madow bias-corrected MI is available and shown next to the plug-in estimate.

## Web interface and static site

A browser interface with the same model and analyses, also published as a static site on GitHub Pages
(https://gddickinson.github.io/ca_signature_analyzer/ once Pages is enabled):

- **Model explorer**: CalciumDynamicsModel with every parameter that enters the equations on a slider; cytosolic,
  ER and vacuolar Ca²⁺ and InsP3 update live, with the signature features.
- **Analyse traces**: upload a CSV (time in s, then one [Ca²⁺] column per trace, nM or μM) or use simulated demo
  traces from the model: signature features, bandwidth, SNR, Shannon capacity, binned entropy, the mutual
  information matrix between traces and steady-state CDPK activation; metrics export as CSV.
- **Signature tables**: the example database, a simulated table (7 stimuli × 5 intensities) or your own CSV:
  summary statistics, pairwise discrimination, k-NN mutual information per feature and Random-Forest accuracy.
- **Format & metrics**: the input formats and the definition of every number.

On the static site everything runs in the browser in a JavaScript port of the Python (`web/static/cs_*.js`),
checked by a node test against values written by the repo's code (`scripts/make_cs_fixture.py`): analyses agree to
1e-9 or better, model trajectories agree with scipy's `odeint` to ~1e-6 relative. Only the scikit-learn estimators
(k-NN mutual information, Random Forest) need the local app; they are shown precomputed for the built-in tables.
Uploaded files are not sent anywhere on the static site.

### Run the web app

```bash
pip install -r requirements.txt -r requirements-web.txt
python run_web.py                    # http://127.0.0.1:8134 (local only)
python -m pytest -q tests/           # includes tests/test_web.py and the node test of the JS port
python scripts/build_static.py       # writes the static site to site/
```

| Model explorer | Trace analysis |
|---|---|
| ![Model explorer](docs/img/model.png) | ![Trace analysis on simulated demo traces](docs/img/analyse_demo.png) |
| **Signature tables** | **Home** |
| ![Signature tables, simulated set](docs/img/signatures_simulated.png) | ![Home page](docs/img/home.png) |

## Project Structure

```
ca_signature_analyzer/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── signature_database.py    # Signature data management
│   ├── biophysical_model.py     # ODE-based Ca²⁺ dynamics
│   ├── decoder_model.py         # CDPK decoder models
│   ├── information_theory.py    # MI and information metrics
│   ├── visualization.py         # Plotting functions
│   ├── utils.py                 # Helper utilities
│   └── gui_app.py              # Main GUI application
├── data/                        # Data directory
│   └── example_signatures.csv   # Example database
├── outputs/                     # Output directory
├── examples/                    # Example scripts
├── docs/                        # Documentation
├── tests/                       # Unit tests
├── run_simulation.py           # Standalone simulation script
├── analyze_database.py         # Standalone analysis script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Core Modules

### signature_database.py

Manages Ca²⁺ signature data:
- `CalciumSignature`: Data class for individual signatures
- `SignatureDatabase`: Database management
- `create_example_database()`: Generate example data

### biophysical_model.py

ODE-based Ca²⁺ dynamics simulation:
- `CalciumDynamicsModel`: Main simulation class
- `ModelParameters`: Model parameters
- Supports 7 stimulus types with distinct signatures

### decoder_model.py

CDPK decoder activation models:
- `CDPKDecoder`: Single decoder model
- `DecoderPanel`: Panel of 10 CDPKs with different affinities
- `PhosphorylationTarget`: Target protein phosphorylation

### information_theory.py

Information-theoretic analyses:
- `InformationAnalyzer`: MI and entropy calculations
- `PathwayInformationFlow`: Multi-stage information flow
- `SignatureDiscriminability`: Classification and discrimination

### visualization.py

Plotting and visualization:
- `SignaturePlotter`: Signature plots, heatmaps, UMAP
- `IntegratedReport`: HTML report generation

## Example Workflows

### Workflow 1: Basic Simulation and Analysis

```python
from src.biophysical_model import CalciumDynamicsModel
from src.decoder_model import DecoderPanel
import matplotlib.pyplot as plt

# Create model
model = CalciumDynamicsModel()

# Simulate ABA response
t, sol = model.simulate('ABA', duration=600, intensity=1.0)
ca_cyt = sol[:, 0]

# Extract features
features = model.extract_signature_features(t, ca_cyt)
print("Signature features:", features)

# Analyze with decoders
panel = DecoderPanel()
responses = panel.analyze_signature(t, ca_cyt)

# Plot
plt.plot(t, ca_cyt / 1000)  # Convert to μM
plt.xlabel('Time (s)')
plt.ylabel('[Ca²⁺] (μM)')
plt.title('ABA-induced Ca²⁺ Signature')
plt.show()
```

### Workflow 2: Database Analysis

```python
from src.signature_database import create_example_database
from src.information_theory import SignatureDiscriminability
import numpy as np

# Load database
db = create_example_database()

# Get features
features = db.get_feature_matrix()
labels = np.array([sig.stimulus for sig in db.signatures])

# Test classification
results = SignatureDiscriminability.classification_accuracy(
    features, labels, classifier='random_forest'
)

print(f"Classification accuracy: {results['accuracy_mean']:.3f}")
print(f"Information transmitted: {results['information_bits']:.3f} bits")
```

### Workflow 3: Custom Signature Addition

```python
from src.signature_database import SignatureDatabase, CalciumSignature

# Create database
db = SignatureDatabase()

# Add custom signature
custom_sig = CalciumSignature(
    signature_id="CUSTOM_001",
    species="Arabidopsis thaliana",
    cell_type="mesophyll",
    stimulus="my_stimulus",
    baseline_ca=100,
    peak_amplitude=1200,
    rise_time=10,
    decay_tau=50,
    duration=200,
    oscillatory=False
)

db.add_signature(custom_sig)

# Save to file
db.save_to_csv("my_signatures.csv")
```

## Output Files

The analyzer generates various output files in the `outputs/` directory:

- **CSV files**: Simulation time series data
- **PNG files**: Plots and visualizations
  - `signature_comparison.png`: Multi-signature overlay
  - `decoder_heatmap.png`: CDPK response heatmap
  - `umap_projection.png`: Feature space visualization
  - `confusion_matrix.png`: Classification results
  - `information_flow.png`: Information flow diagram
- **JSON files**: Analysis results and statistics
- **HTML files**: Comprehensive reports

## Testing

Run the unit tests:

```bash
python -m pytest tests/
```

Run module self-tests:

```bash
python src/signature_database.py
python src/biophysical_model.py
python src/decoder_model.py
python src/information_theory.py
python src/visualization.py
python src/utils.py
```

## Advanced Configuration

### Custom Model Parameters

Modify biophysical model parameters:

```python
from src.biophysical_model import CalciumDynamicsModel, ModelParameters

# Create custom parameters
params = ModelParameters(
    V_PM_GLR=100.0,  # Increase channel activity
    K_pump=150.0,    # Change pump affinity
    # ... other parameters
)

# Use custom model
model = CalciumDynamicsModel(params)
t, sol = model.simulate('ABA', duration=600)
```

### Custom CDPK Parameters

Add custom CDPK decoders:

```python
from src.decoder_model import CDPKParameters, DecoderPanel, CDPKDecoder

# Define custom CDPK
custom_cdpk = CDPKParameters(
    name="CUSTOM_CPK",
    K_d=500.0,      # 500 nM affinity
    n_Hill=3.0,     # Cooperativity
    k_on=1e8,
    k_off=50.0,
    k_cat=12.0
)

# Create decoder
decoder = CDPKDecoder(custom_cdpk)

# Or add to panel
panel = DecoderPanel([custom_cdpk])
```

## Troubleshooting

### Common Issues

**Import errors**: Ensure all dependencies are installed
```bash
pip install -r requirements.txt
```

**GUI doesn't launch**: Check that tkinter is installed (usually comes with Python)
```bash
python -m tkinter  # Should open a test window
```

**UMAP errors**: Install umap-learn
```bash
pip install umap-learn
```

**Plotting errors**: Ensure matplotlib backend is properly configured
```python
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg'
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

## Acknowledgments

This project was developed as part of research on plant calcium signaling and
information theory analysis.

## Version History

- **v1.0.0** (2025): Initial release
  - Core modules implemented
  - GUI application
  - Standalone scripts
  - Example database
  - Comprehensive documentation


---
*Built with AI assistance from [Claude (Anthropic)](https://claude.com/).*
