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


# ── pass-through when n < 300 ──────────────────────────────────────────────────

def test_transform_is_passthrough_when_fewer_than_300_samples():
    cal = Calibrator()
    raw, outcomes = _synthetic_data(n=299)
    cal.fit(raw, outcomes)
    for p in [0.1, 0.5, 0.9]:
        assert cal.transform(p) == pytest.approx(p)


def test_transform_calibrates_when_exactly_300_samples():
    cal = Calibrator()
    raw, outcomes = _synthetic_data(n=300)
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


# ── Phase 2 new tests ──────────────────────────────────────────────────────────

def test_fit_uses_y_up_labels():
    """y_up labels should produce different calibration than inverted labels."""
    cal_correct = Calibrator()
    cal_inverted = Calibrator()
    raw = np.array([0.1, 0.9] * 200)
    y_up = np.array([0, 1] * 200, dtype=float)
    y_inverted = np.array([1, 0] * 200, dtype=float)
    cal_correct.fit(raw, y_up)
    cal_inverted.fit(raw, y_inverted)
    # Calibrated outputs for same raw input should differ
    assert cal_correct.transform(0.7) != cal_inverted.transform(0.7)


def test_min_samples_is_300():
    """Passthrough should be True for n=299, False for n=300."""
    cal_under = Calibrator()
    raw, outcomes = _synthetic_data(n=299)
    cal_under.fit(raw, outcomes)
    assert cal_under._passthrough is True

    cal_at = Calibrator()
    raw, outcomes = _synthetic_data(n=300)
    cal_at.fit(raw, outcomes)
    assert cal_at._passthrough is False


def test_monotonicity_guard_reverts_worse_fit():
    """After a clean fit, a contradictory fit that worsens Brier should be reverted."""
    cal = Calibrator()
    # First fit: clean data — calibrator engages
    raw_clean, outcomes_clean = _synthetic_data(n=400, seed=1)
    cal.fit(raw_clean, outcomes_clean)
    model_after_first = cal._model
    brier_after_first = cal._prev_brier

    # Second fit: perfectly contradictory (high raw → low outcome) → Brier worsens
    raw_bad = np.linspace(0.1, 0.9, 400)
    outcomes_bad = (raw_bad < 0.5).astype(float)  # inverted correlation → bad calibration
    cal.fit(raw_bad, outcomes_bad)

    # If Brier worsened, the model should have been reverted
    if brier_after_first is not None:
        assert cal._model is model_after_first or cal._prev_brier <= brier_after_first + 1e-6


def test_save_load_with_correct_labels():
    """Save a calibrator fit with y_up labels; load and verify transform matches."""
    cal = Calibrator()
    raw = np.array([0.1, 0.9] * 200)
    y_up = np.array([0, 1] * 200, dtype=float)
    cal.fit(raw, y_up)
    expected = cal.transform(0.7)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "cal_y_up.pkl")
        cal.save(path)
        cal2 = Calibrator.load(path)
        assert cal2.transform(0.7) == pytest.approx(expected, abs=1e-9)


# ── quadratic logistic calibrator (Phase 3) ───────────────────────────────────

def test_calibrator_uses_logistic_regression():
    """Calibrator should use LogisticRegression, not IsotonicRegression."""
    from sklearn.linear_model import LogisticRegression
    cal = Calibrator()
    raw, outcomes = _synthetic_data(n=500)
    cal.fit(raw, outcomes)
    assert isinstance(cal._model, LogisticRegression)


def test_inverted_signal_calibrated_below_half():
    """High raw→low outcome (k15_raw>0.8 = P(up)=0.38) should calibrate below 0.5."""
    rng = np.random.default_rng(7)
    n = 600
    raw = rng.uniform(0.0, 1.0, n)
    # Inverted-U: high raw (>0.7) → mostly outcome=0; mid range → mostly outcome=1
    p_outcome = np.where(raw > 0.7, 0.15, np.where(raw < 0.3, 0.25, 0.75))
    outcomes = (rng.uniform(0, 1, n) < p_outcome).astype(float)
    cal = Calibrator()
    cal.fit(raw, outcomes)
    assert cal.transform(0.9) < 0.5


def test_passthrough_still_works_below_min_samples_logistic():
    """Passthrough is preserved below _MIN_SAMPLES with the logistic calibrator."""
    cal = Calibrator()
    raw, outcomes = _synthetic_data(n=150)
    cal.fit(raw, outcomes)
    assert cal._passthrough is True
    for p in [0.1, 0.5, 0.9]:
        assert cal.transform(p) == pytest.approx(p)
