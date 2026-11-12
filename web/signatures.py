"""Signature-table analyses for the web interface (SignatureDatabase + information_theory).

- `example_database()`: the repo's literature-derived example database (create_example_database).
- `simulated_set()`: a SIMULATED demo table: CalciumDynamicsModel runs (7 stimuli x 5 intensities, 600 s, dt 0.1 s),
  converted to CalciumSignature rows the way the GUI's "Save simulation to database" does.
- `analyse_table(db, full)`: summary statistics, feature matrix, pairwise discrimination and (full=True, Python only)
  estimate_signature_capacity and Random-Forest classification_accuracy.
"""
import io
import warnings

import numpy as np

from web.analysis import clean
from biophysical_model import CalciumDynamicsModel  # noqa: E402
from information_theory import InformationAnalyzer, SignatureDiscriminability  # noqa: E402
from signature_database import CalciumSignature, SignatureDatabase, create_example_database  # noqa: E402

FEATURES = ["baseline_ca", "peak_amplitude", "fold_change", "rise_time", "decay_tau", "duration", "latency",
            "frequency", "duty_cycle"]
DEMO_INTENSITIES = [0.5, 0.75, 1.0, 1.25, 1.5]
_cache = {}


def example_database():
    return create_example_database()


def load_csv_text(text):
    db = SignatureDatabase()
    db.load_from_csv(io.StringIO(text))
    if len(db) == 0:
        raise ValueError("No signatures in the file.")
    return db


def simulated_set():
    if "sim" in _cache:
        return _cache["sim"]
    model, db = CalciumDynamicsModel(), SignatureDatabase()
    nz = (lambda v: 0 if v is None or np.isnan(v) else v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for stim in CalciumDynamicsModel.VALID_STIMULI:
            for k, inten in enumerate(DEMO_INTENSITIES):
                t, sol = model.simulate(stim, duration=600.0, dt=0.1, intensity=inten)
                f = model.extract_signature_features(t, sol[:, 0])
                db.add_signature(CalciumSignature(
                    signature_id=f"{stim}_I{inten:g}", species="Arabidopsis thaliana (simulated)",
                    cell_type="model", stimulus=stim, stimulus_concentration=None,
                    baseline_ca=f["baseline_ca"], peak_amplitude=f["peak_amplitude"], rise_time=nz(f["rise_time"]),
                    decay_tau=nz(f["decay_tau"]), duration=f["duration"], latency=nz(f["latency"]),
                    oscillatory=f["oscillatory"], frequency=f["frequency"],
                    notes=f"simulated: CalciumDynamicsModel, intensity {inten:g}"))
    _cache["sim"] = db
    return db


def _rows(db):
    rows = []
    for s in db.signatures:
        d = s.to_dict()
        d["fold_change"] = s.fold_change
        d["snr"] = s.snr
        rows.append(d)
    return rows


def analyse_table(db, full=True):
    X = db.get_feature_matrix()
    labels = np.array([s.stimulus for s in db.signatures])
    classes, counts = np.unique(labels, return_counts=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = db.get_summary_statistics()
        disc = SignatureDiscriminability.pairwise_discrimination(X, labels)
    out = {"rows": _rows(db), "feature_names": FEATURES, "features": X, "summary": stats,
           "classes": list(classes), "class_counts": counts, "discrimination": disc}
    if full:
        out["python"] = python_only(X, labels, classes, counts)
    return clean(out)


def python_only(X, labels, classes, counts):
    """Estimators with no JS port: sklearn's k-NN MI (mutual_info_classif) and a Random Forest."""
    res = {}
    codes = np.searchsorted(classes, labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            if len(classes) < 2:
                raise ValueError("needs at least two stimulus classes")
            res["capacity"] = InformationAnalyzer.estimate_signature_capacity(X, codes)
        except Exception as e:  # noqa: BLE001
            res["capacity_error"] = str(e)
        try:
            if len(classes) < 2:
                raise ValueError("needs at least two stimulus classes")
            r = SignatureDiscriminability.classification_accuracy(X, labels, classifier="random_forest")
            res["classification"] = {k: r[k] for k in ("accuracy_mean", "accuracy_std", "cv_scores",
                                                       "confusion_matrix", "n_classes", "n_samples",
                                                       "information_bits", "max_information_bits")}
        except Exception as e:  # noqa: BLE001
            res["classification_error"] = str(e)
    return res
