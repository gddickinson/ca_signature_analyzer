"""
Tests for src.signature_database: CSV round trip and the feature matrix.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from signature_database import SignatureDatabase, create_example_database  # noqa: E402

EXAMPLE = Path(__file__).parent.parent / "data" / "example_signatures.csv"


def test_loaded_example_matches_built_in():
    """data/example_signatures.csv loads to the same feature matrix as create_example_database()."""
    db = SignatureDatabase()
    db.load_from_csv(str(EXAMPLE))
    built = create_example_database()
    assert len(db) == len(built) == 7
    np.testing.assert_allclose(db.get_feature_matrix(), built.get_feature_matrix())


def test_missing_values_load_as_none():
    """Empty CSV cells become None (not NaN), so the feature matrix stays finite."""
    db = SignatureDatabase()
    db.load_from_csv(str(EXAMPLE))
    nacl = db.filter_by_stimulus("NaCl")[0]
    assert nacl.frequency is None and nacl.duty_cycle is None and nacl.notes is None
    assert np.all(np.isfinite(db.get_feature_matrix()))


def test_missing_oscillatory_is_false(tmp_path):
    """An empty 'oscillatory' cell means not oscillatory (bool(NaN) is True)."""
    csv = tmp_path / "sigs.csv"
    csv.write_text("signature_id,species,cell_type,stimulus,baseline_ca,peak_amplitude,rise_time,"
                   "decay_tau,duration,oscillatory\nX1,A,leaf,NaCl,100,600,30,300,600,\n")
    db = SignatureDatabase()
    db.load_from_csv(str(csv))
    assert db[0].oscillatory is False
    assert db[0].latency == pytest.approx(5.0)


def test_example_frequencies_are_in_hz():
    """Example oscillation frequencies were per-minute values labelled Hz (0.5 'Hz' = a 2 s period).
    Stored in Hz: plant Ca2+ oscillations have periods of minutes, and the model's H2O2
    oscillation (0.4 per minute = 0.4/60 Hz) matches the H2O2 example."""
    db = create_example_database()
    freqs = {s.stimulus: s.frequency for s in db.signatures if s.frequency is not None}
    assert freqs["ABA"] == pytest.approx(0.5 / 60)
    assert freqs["flg22"] == pytest.approx(0.2 / 60)
    assert freqs["H2O2"] == pytest.approx(0.4 / 60)
    loaded = SignatureDatabase()
    loaded.load_from_csv(str(EXAMPLE))
    assert {s.stimulus: s.frequency for s in loaded.signatures if s.frequency is not None} == pytest.approx(freqs)
