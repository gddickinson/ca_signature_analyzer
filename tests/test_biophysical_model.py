"""
Tests for src.biophysical_model module.

Tests ODE model simulation and feature extraction against known behavior.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biophysical_model import CalciumDynamicsModel, ModelParameters


class TestModelParameters:
    """Tests for ModelParameters dataclass."""

    def test_defaults(self):
        params = ModelParameters()
        assert params.V_PM_GLR > 0
        assert params.K_pump > 0
        assert params.B_total > 0

    def test_custom_params(self):
        params = ModelParameters(V_PM_GLR=100.0)
        assert params.V_PM_GLR == 100.0


class TestCalciumDynamicsModel:
    """Tests for CalciumDynamicsModel."""

    def test_creation(self):
        model = CalciumDynamicsModel()
        assert model.params is not None
        assert len(model.state_names) == 4

    def test_simulate_returns_correct_shapes(self):
        model = CalciumDynamicsModel()
        t, sol = model.simulate("ABA", duration=60, dt=0.5)
        assert t.shape[0] == sol.shape[0]
        assert sol.shape[1] == 4  # 4 state variables

    def test_simulate_all_stimuli(self):
        """All stimulus types should run without error."""
        model = CalciumDynamicsModel()
        stimuli = ["ABA", "NaCl", "mechanical_touch", "flg22",
                   "cold_shock", "H2O2", "glutamate"]
        for stim in stimuli:
            t, sol = model.simulate(stim, duration=30, dt=1.0)
            assert sol.shape[0] > 0
            # Ca_cyt should stay non-negative
            assert np.all(sol[:, 0] >= -1.0)  # Allow tiny numerical error

    def test_resting_state(self):
        """With no stimulus, Ca should stay near baseline."""
        model = CalciumDynamicsModel()
        # Use a stimulus type that has a long latency
        t, sol = model.simulate("ABA", duration=30, dt=0.5)
        ca_cyt_early = sol[:10, 0]  # First few points before stimulus kicks in
        # Should stay near resting level (100 nM)
        assert np.all(ca_cyt_early < 500)

    def test_stimulus_response(self):
        """Stimulus should cause Ca increase for strong stimuli."""
        model = CalciumDynamicsModel()
        t, sol = model.simulate("NaCl", duration=120, dt=0.5, intensity=2.0)
        ca_cyt = sol[:, 0]
        baseline = ca_cyt[0]
        peak = np.max(ca_cyt)
        # NaCl should cause a measurable response
        assert peak > baseline

    def test_custom_initial_state(self):
        model = CalciumDynamicsModel()
        y0 = np.array([200.0, 40000.0, 200000.0, 5.0])
        t, sol = model.simulate("ABA", duration=10, dt=1.0, initial_state=y0)
        assert sol[0, 0] == pytest.approx(200.0, abs=1.0)

    def test_negative_duration_produces_empty(self):
        """Negative duration should produce empty or single-point array."""
        model = CalciumDynamicsModel()
        t, sol = model.simulate("ABA", duration=-1, dt=0.5)
        assert len(t) == 0 or len(t) == 1


class TestExtractSignatureFeatures:
    """Tests for feature extraction."""

    def test_basic_features(self):
        model = CalciumDynamicsModel()
        t, sol = model.simulate("NaCl", duration=120, dt=0.5)
        ca_cyt = sol[:, 0]
        features = model.extract_signature_features(t, ca_cyt)
        assert "baseline_ca" in features
        assert "peak_amplitude" in features
        assert features["baseline_ca"] > 0

    def test_flat_trace(self):
        """A flat trace should return no significant features."""
        model = CalciumDynamicsModel()
        t = np.linspace(0, 100, 200)
        ca_flat = np.full_like(t, 100.0)
        features = model.extract_signature_features(t, ca_flat)
        assert features["duration"] == 0


class TestFeatureRegressions:
    """Regressions for extract_signature_features (see SESSION_LOG.md)."""

    KEYS = {"baseline_ca", "peak_amplitude", "rise_time", "decay_tau", "duration",
            "latency", "oscillatory", "frequency", "n_peaks"}

    def test_flat_trace_has_all_keys(self):
        """A flat trace returns the same keys as a responding one (the GUI reads 'oscillatory')."""
        model = CalciumDynamicsModel()
        t = np.linspace(0, 100, 200)
        features = model.extract_signature_features(t, np.full_like(t, 100.0))
        assert set(features) == self.KEYS
        assert features["oscillatory"] is False and features["n_peaks"] == 0
        assert features["frequency"] is None

    def test_min_peak_spacing_is_5s(self):
        """Peaks 5.0 s apart are all counted; peaks 4.9 s apart are not (5 // 0.1 == 49 samples was used)."""
        model = CalciumDynamicsModel()
        dt = 0.1
        t = np.arange(0, 60, dt)
        ca = np.full_like(t, 100.0)
        for tp in (20.0, 25.0, 30.0):
            ca += 900.0 * np.exp(-0.5 * ((t - tp) / 0.3) ** 2)
        features = model.extract_signature_features(t, ca)
        assert features["n_peaks"] == 3
        assert features["frequency"] == pytest.approx(0.2, rel=1e-9)
        close = np.full_like(t, 100.0)
        for tp in (20.0, 24.9, 29.8):
            close += 900.0 * np.exp(-0.5 * ((t - tp) / 0.3) ** 2)
        assert model.extract_signature_features(t, close)["n_peaks"] < 3

    def test_coarse_sampling_does_not_crash(self):
        """With samples more than 5 s apart the peak spacing is one sample, not 0 (scipy raises on 0)."""
        model = CalciumDynamicsModel()
        t = np.arange(0, 600, 6.0)
        ca = 100.0 + 500.0 * (np.sin(2 * np.pi * t / 60.0) > 0.9)
        features = model.extract_signature_features(t, ca)
        assert features["n_peaks"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
