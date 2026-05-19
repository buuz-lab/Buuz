import os
import tempfile

import numpy as np
import pytest

from btc_kalshi_system.models.calibrator import Calibrator


def _synthetic_data(n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, n)
    # outcomes correlated with raw probs so calibration actually changes something
    outcomes = (rng.uniform(0, 1, n) < raw).astype(int)
    return raw, outcomes


# ── fit / transform ────────────────────────────────────────────────────────────

def test_transform_returns_float_after_fit():
    cal = Calibrator()
    raw, outcomes = _synthetic_data()
    cal.fit(raw, outcomes)
    result = cal.transform(0.6)
    assert isinstance(result, float)


def test_transform_output_in_unit_interval():
    cal = Calibrator()
    raw, outcomes = _synthetic_data()
    cal.fit(raw, outcomes)
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        assert 0.0 <= cal.transform(p) <= 1.0


def test_calibrated_output_differs_from_raw_after_fit():
    cal = Calibrator()
    raw, outcomes = _synthetic_data(n=2000)
    cal.fit(raw, outcomes)
    # Isotonic regression should move at least some probabilities
    diffs = [abs(cal.transform(float(p)) - float(p)) for p in np.linspace(0.1, 0.9, 9)]
    assert max(diffs) > 1e-6


# ── pass-through when n < 500 ──────────────────────────────────────────────────

def test_transform_is_passthrough_when_fewer_than_500_samples():
    cal = Calibrator()
    raw, outcomes = _synthetic_data(n=499)
    cal.fit(raw, outcomes)
    for p in [0.1, 0.5, 0.9]:
        assert cal.transform(p) == pytest.approx(p)


def test_transform_calibrates_when_exactly_500_samples():
    cal = Calibrator()
    raw, outcomes = _synthetic_data(n=500)
    cal.fit(raw, outcomes)
    # Should not raise and should return a float — calibration engaged
    result = cal.transform(0.5)
    assert isinstance(result, float)


# ── brier_score ────────────────────────────────────────────────────────────────

def test_brier_score_perfect_predictions_is_zero():
    cal = Calibrator()
    raw = np.array([1.0, 1.0, 0.0, 0.0])
    outcomes = np.array([1, 1, 0, 0])
    cal.fit(raw, outcomes)
    # In pass-through mode (n=4 < 500), brier is computed on raw probs
    assert cal.brier_score(raw, outcomes) == pytest.approx(0.0, abs=1e-9)


def test_brier_score_worst_predictions_is_one():
    cal = Calibrator()
    raw = np.array([0.0, 0.0, 1.0, 1.0])
    outcomes = np.array([1, 1, 0, 0])
    cal.fit(raw, outcomes)
    assert cal.brier_score(raw, outcomes) == pytest.approx(1.0, abs=1e-9)


def test_brier_score_random_guess_is_near_quarter():
    cal = Calibrator()
    raw = np.full(4, 0.5)
    outcomes = np.array([1, 0, 1, 0])
    cal.fit(raw, outcomes)
    # BS = mean((0.5-y)^2) = 0.25 for all y in {0,1}
    assert cal.brier_score(raw, outcomes) == pytest.approx(0.25, abs=1e-9)


# ── save / load ────────────────────────────────────────────────────────────────

def test_save_and_load_roundtrip_produces_same_output():
    cal = Calibrator()
    raw, outcomes = _synthetic_data(n=1000)
    cal.fit(raw, outcomes)
    expected = cal.transform(0.7)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "calibrator.joblib")
        cal.save(path)
        cal2 = Calibrator.load(path)
        assert cal2.transform(0.7) == pytest.approx(expected, abs=1e-9)


def test_load_from_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        Calibrator.load("/tmp/does_not_exist_calibrator.joblib")
