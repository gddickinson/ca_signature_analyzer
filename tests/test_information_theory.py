"""
Tests for src.information_theory module.

Tests information-theoretic calculations with synthetic data.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from information_theory import (InformationAnalyzer, PathwayInformationFlow,
                                SignatureDiscriminability)


class TestInformationAnalyzer:
    """Tests for InformationAnalyzer static methods."""

    def test_mutual_information_discrete_identical(self):
        """MI of identical variables should equal entropy."""
        x = np.array([0, 0, 1, 1, 2, 2])
        mi = InformationAnalyzer.mutual_information_discrete(x, x)
        h = InformationAnalyzer.entropy_discrete(x)
        assert mi == pytest.approx(h, abs=0.01)

    def test_mutual_information_discrete_independent(self):
        """MI of independent variables should be near zero."""
        np.random.seed(42)
        x = np.random.randint(0, 3, size=1000)
        y = np.random.randint(0, 3, size=1000)
        mi = InformationAnalyzer.mutual_information_discrete(x, y)
        assert mi < 0.1  # Should be close to zero

    def test_entropy_uniform(self):
        """Entropy of uniform distribution over N values = log2(N)."""
        # 4 equally likely values
        x = np.array([0, 1, 2, 3] * 100)
        h = InformationAnalyzer.entropy_discrete(x)
        assert h == pytest.approx(2.0, abs=0.01)  # log2(4) = 2

    def test_entropy_deterministic(self):
        """Entropy of a deterministic variable is 0."""
        x = np.array([1, 1, 1, 1, 1])
        h = InformationAnalyzer.entropy_discrete(x)
        assert h == pytest.approx(0.0, abs=0.01)

    def test_channel_capacity_shannon(self):
        """Known Shannon-Hartley values."""
        # C = W * log2(1 + SNR)
        capacity = InformationAnalyzer.channel_capacity_shannon(1.0, 1.0)
        assert capacity == pytest.approx(1.0, abs=0.01)  # log2(2) = 1

        capacity = InformationAnalyzer.channel_capacity_shannon(1.0, 3.0)
        assert capacity == pytest.approx(2.0, abs=0.01)  # log2(4) = 2

    def test_mutual_information_continuous(self):
        """Correlated variables should have higher MI than independent."""
        np.random.seed(42)
        x = np.random.randn(200)
        y_corr = x + 0.1 * np.random.randn(200)
        y_indep = np.random.randn(200)

        mi_corr = InformationAnalyzer.mutual_information_continuous(x, y_corr)
        mi_indep = InformationAnalyzer.mutual_information_continuous(x, y_indep)
        assert mi_corr > mi_indep

    def test_estimate_signature_capacity(self):
        """Capacity estimation should return expected keys."""
        np.random.seed(42)
        # 3 classes, well-separated
        sigs = np.vstack([
            np.random.randn(20, 3) + i * 5 for i in range(3)
        ])
        labels = np.array([0]*20 + [1]*20 + [2]*20)
        metrics = InformationAnalyzer.estimate_signature_capacity(sigs, labels)
        assert "n_stimuli" in metrics
        assert metrics["n_stimuli"] == 3
        assert metrics["max_information_bits"] == pytest.approx(np.log2(3), abs=0.01)


class TestSignatureDiscriminability:
    """Tests for SignatureDiscriminability."""

    def test_pairwise_discrimination_shape(self):
        np.random.seed(42)
        sigs = np.vstack([np.random.randn(10, 3) + i * 5 for i in range(3)])
        labels = np.array([0]*10 + [1]*10 + [2]*10)
        matrix = SignatureDiscriminability.pairwise_discrimination(sigs, labels)
        assert matrix.shape == (3, 3)
        # Diagonal should be zero
        assert np.all(np.diag(matrix) == 0)
        # Off-diagonal should be positive
        assert np.all(matrix[matrix != 0] > 0)

    def test_classification_accuracy_keys(self):
        np.random.seed(42)
        sigs = np.vstack([np.random.randn(20, 3) + i * 5 for i in range(3)])
        labels = np.array([0]*20 + [1]*20 + [2]*20)
        results = SignatureDiscriminability.classification_accuracy(sigs, labels)
        assert "accuracy_mean" in results
        assert "confusion_matrix" in results
        # Well-separated classes should be classified with high accuracy
        assert results["accuracy_mean"] > 0.8


class TestPathwayInformationFlow:
    """Regression: analyze_database.py and the GUI pass stimulus names (strings) as the first stage."""

    def _flow(self, stimulus_stage):
        rng = np.random.default_rng(0)
        codes = np.repeat(np.arange(4), 5)
        features = codes[:, None] * 3.0 + rng.normal(size=(20, 3))
        decoders = features @ rng.normal(size=(3, 5))
        flow = PathwayInformationFlow()
        flow.add_stage("Stimulus", stimulus_stage)
        flow.add_stage("Ca_Signature", features)
        flow.add_stage("CDPK_Response", decoders)
        return flow.calculate_flow()

    def test_text_labels_match_integer_codes(self):
        names = np.array(["ABA", "NaCl", "cold_shock", "flg22"])
        text = self._flow(np.repeat(names, 5).reshape(-1, 1))
        codes = self._flow(np.repeat(np.arange(4), 5).reshape(-1, 1))
        assert text.keys() == codes.keys()
        for key in codes:
            assert text[key] == pytest.approx(codes[key])


class TestCapacityAndLabels:
    """Regressions found running analyze_database.py on the example database."""

    def test_capacity_needs_two_signatures_per_stimulus(self):
        """One signature per stimulus: a clear ValueError, not sklearn's 'Found array with 0 sample(s)'."""
        sigs = np.arange(21, dtype=float).reshape(7, 3)
        with pytest.raises(ValueError, match="at least 2 signatures per stimulus"):
            InformationAnalyzer.estimate_signature_capacity(sigs, np.arange(7))

    def test_cli_label_order_matches_confusion_matrix(self):
        """analyze_database labels the confusion matrix in the sorted order sklearn uses for its rows."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import analyze_database
        labels = ["flg22", "ABA", "NaCl", "ABA", "flg22"]
        names, codes = analyze_database.encode_stimuli(labels)
        assert names == ["ABA", "NaCl", "flg22"]
        assert list(codes) == [2, 0, 1, 0, 2]


class TestEstimatorFixes:
    """Regressions for the information estimators (see SESSION_LOG.md)."""

    def test_flow_initial_entropy_uses_the_mi_binning(self):
        """Four equiprobable stimuli carry log2(4) = 2 bits (quantile binning merged categories)."""
        rng = np.random.default_rng(1)
        flow = PathwayInformationFlow()
        labels = np.repeat(["a", "b", "c", "d"], 5).reshape(-1, 1)
        flow.add_stage("Stimulus", labels)
        flow.add_stage("Features", rng.normal(size=(20, 3)))
        out = flow.calculate_flow()
        assert out["initial_entropy"] == pytest.approx(2.0)

    def test_miller_madow_entropy(self):
        x = np.array([0, 0, 1, 1])
        expected = 1.0 + (2 - 1) / (2 * 4 * np.log(2))
        assert InformationAnalyzer.entropy_miller_madow(x) == pytest.approx(expected)

    def test_miller_madow_mi_reduces_bias(self):
        """Independent variables: the plug-in MI is biased upward; Miller-Madow is much closer to 0."""
        rng = np.random.default_rng(5)
        plug, mm = [], []
        for _ in range(20):
            x, y = rng.integers(0, 6, 60), rng.integers(0, 6, 60)
            plug.append(InformationAnalyzer.mutual_information_discrete(x, y))
            mm.append(InformationAnalyzer.mutual_information_discrete(x, y, bias_correction="miller_madow"))
        assert np.mean(plug) > 0.2
        assert abs(np.mean(mm)) < 0.5 * np.mean(plug)

    def test_classification_information_is_mi_of_confusion_matrix(self):
        """Chance-level classification transmits ~0 bits (log2(N) x accuracy gave ~0.5 bits)."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict
        rng = np.random.default_rng(2)
        sigs = rng.normal(size=(90, 3))
        labels = np.repeat([0, 1, 2], 30)
        res = SignatureDiscriminability.classification_accuracy(sigs, labels)
        pred = cross_val_predict(RandomForestClassifier(n_estimators=100, random_state=42), sigs, labels, cv=5)
        assert res["information_bits"] == pytest.approx(InformationAnalyzer.mutual_information_discrete(labels, pred))
        assert res["information_bits"] < 0.2
        assert res["information_bits_miller_madow"] < res["information_bits"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
