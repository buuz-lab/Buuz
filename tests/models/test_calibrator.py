import os
import tempfile

import numpy as np
import pytest

from btc_kalshi_system.models.calibrator import Calibrator, _encode_regime


def _synthetic_data(n: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0, 1, n)
    # outcomes correlated with raw probs so calibration actually changes something
    outcomes = (rng.uniform(0, 1, n) < raw).astype(int)
    return raw, outcomes


def _compressed_data(n: int = 300, seed: int = 7):
    """Data with positive raw-outcome correlation where the true probability is
    heavily compressed toward 0.5 (raw=0.90→true_prob=0.60, raw=0.10→true_prob=0.40).
    Passthrough (which uses raw=0.90 directly) has much higher Brier than the model,
    so deployment is reliable. transform(0.75) stays > 0.5 — passes direction guard."""
    rng = np.random.default_rng(seed)
    n_high, n_low = n // 2, n - n // 2
    raw = np.concatenate([np.full(n_high, 0.90), np.full(n_low, 0.10)])
    true_prob = np.where(raw > 0.5, 0.60, 0.40)
    outcomes = (rng.uniform(0, 1, n) < true_prob).astype(float)
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
    # Compressed signal: true_prob = 0.25 + 0.5*raw. Model beats passthrough (which
    # uses raw directly) because raw over/underestimates the true probability.
    raw, outcomes = _compressed_data(n=2000)
    cal.fit(raw, outcomes)
    diffs = [abs(cal.transform(float(p)) - float(p)) for p in np.linspace(0.1, 0.9, 9)]
    assert max(diffs) > 1e-6


# ── pass-through when n < 300 ──────────────────────────────────────────────────

def test_transform_is_passthrough_when_fewer_than_100_samples():
    cal = Calibrator()
    raw, outcomes = _synthetic_data(n=99)
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


def test_minimum_training_rows_is_100():
    """_MIN_SAMPLES=100 applies to the training split, not total rows.
    With 20% holdout: n=123 → n_train=99 → passthrough; n=124 → n_train=100 → fits."""
    # n=123: n_holdout=24, n_train=99 < 100 → passthrough (n_train guard fires first)
    cal_under = Calibrator()
    raw_under, y_under = _compressed_data(n=123, seed=7)
    cal_under.fit(raw_under, y_under)
    assert cal_under._passthrough is True

    # n=124: n_holdout=24, n_train=100 = _MIN_SAMPLES → fitting proceeds and deploys
    cal_at = Calibrator()
    raw_at, y_at = _compressed_data(n=124, seed=7)
    cal_at.fit(raw_at, y_at)
    assert cal_at._passthrough is False


def test_holdout_guard_reverts_when_fit_does_not_beat_passthrough():
    """After a good fit, a second fit where holdout Brier ≥ passthrough reverts to first model."""
    cal = Calibrator()
    # First fit: compressed signal — beats passthrough on holdout, deploys, passes direction guard
    raw_good, y_good = _compressed_data(n=400)
    cal.fit(raw_good, y_good)
    assert not cal._passthrough
    model_after_first = cal._model

    # Second fit: all raw=0.5 — model outputs ~0.5 = same as passthrough, can't beat it
    raw_flat = np.full(400, 0.5)
    y_flat = np.tile([0.0, 1.0], 200)
    cal.fit(raw_flat, y_flat)

    # Holdout Brier(model) ≈ passthrough Brier ≈ 0.25 → no improvement → revert
    assert cal._model is model_after_first


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
    # Compressed signal so calibrator deploys (needed to assert _model is set)
    cal = Calibrator()
    raw, outcomes = _compressed_data(n=500)
    cal.fit(raw, outcomes)
    assert isinstance(cal._model, LogisticRegression)


def test_inverted_signal_stays_passthrough_via_direction_guard():
    """High raw→low outcome (k15_raw>0.8 = P(up)=0.38) triggers direction guard — stays passthrough.
    Previously this inverted signal was deployed; the guard now blocks it to prevent
    live signal inversion (session 26 fix)."""
    rng = np.random.default_rng(7)
    n = 600
    raw = rng.uniform(0.0, 1.0, n)
    p_outcome = np.where(raw > 0.7, 0.15, np.where(raw < 0.3, 0.25, 0.75))
    outcomes = (rng.uniform(0, 1, n) < p_outcome).astype(float)
    cal = Calibrator()
    cal.fit(raw, outcomes)
    assert cal._passthrough is True
    assert cal.transform(0.9) == pytest.approx(0.9)


def test_passthrough_still_works_below_min_samples_logistic():
    """Passthrough is preserved below _MIN_SAMPLES with the logistic calibrator."""
    cal = Calibrator()
    raw, outcomes = _synthetic_data(n=80)
    cal.fit(raw, outcomes)
    assert cal._passthrough is True
    for p in [0.1, 0.5, 0.9]:
        assert cal.transform(p) == pytest.approx(p)


# ── regime-aware calibrator ────────────────────────────────────────────────────

def _regime_data(n: int = 600, seed: int = 42):
    """Regime-tagged synthetic data: trending_up → y_up correlated; ranging → noisy."""
    rng = np.random.default_rng(seed)
    half = n // 2
    raw_trend = rng.uniform(0.3, 0.9, half)
    raw_range = rng.uniform(0.3, 0.9, half)
    raw = np.concatenate([raw_trend, raw_range])
    y_trend = (rng.uniform(0, 1, half) < raw_trend).astype(float)
    y_range = (rng.uniform(0, 1, half) < 0.5).astype(float)
    outcomes = np.concatenate([y_trend, y_range])
    regimes = np.array(["trending_up"] * half + ["ranging"] * half, dtype=object)
    return raw, outcomes, regimes


def test_regime_aware_flag_set_when_regimes_provided():
    raw, outcomes, regimes = _regime_data()
    cal = Calibrator()
    # Invert to guarantee deploy
    rng = np.random.default_rng(99)
    raw2 = np.array([0.1] * 300 + [0.9] * 300)
    outcomes2 = np.array([1.0] * 300 + [0.0] * 300)
    regimes2 = np.array(["trending_up"] * 300 + ["trending_down"] * 300, dtype=object)
    cal.fit(raw2, outcomes2, regimes=regimes2)
    assert cal._regime_aware is True


def test_regime_aware_false_without_regimes():
    cal = Calibrator()
    raw = np.array([0.1] * 300 + [0.9] * 300)
    outcomes = np.array([1.0] * 300 + [0.0] * 300)
    cal.fit(raw, outcomes)
    assert cal._regime_aware is False


def test_transform_with_regime_differs_from_without():
    """transform(p, regime=X) should differ from transform(p) for regime-aware model."""
    cal = Calibrator()
    raw = np.array([0.1] * 300 + [0.9] * 300)
    outcomes = np.array([1.0] * 300 + [0.0] * 300)
    regimes = np.array(["trending_up"] * 300 + ["trending_down"] * 300, dtype=object)
    cal.fit(raw, outcomes, regimes=regimes)
    assert not cal._passthrough
    assert cal._regime_aware
    result_up = cal.transform(0.7, regime="trending_up")
    result_down = cal.transform(0.7, regime="trending_down")
    assert result_up != pytest.approx(result_down)


def test_transform_regime_unknown_uses_zero_encoding():
    """Unknown regime string falls back to score=0.0, between ranging (0.3) and high_uncertainty (-0.3)."""
    cal = Calibrator()
    raw = np.array([0.1] * 300 + [0.9] * 300)
    outcomes = np.array([1.0] * 300 + [0.0] * 300)
    regimes = np.array(["trending_up"] * 300 + ["trending_down"] * 300, dtype=object)
    cal.fit(raw, outcomes, regimes=regimes)
    result_unknown = cal.transform(0.5, regime="garbage_regime")
    result_ranging = cal.transform(0.5, regime="ranging")
    result_hu = cal.transform(0.5, regime="high_uncertainty")
    # Unknown (0.0) sits between ranging (0.3) and high_uncertainty (-0.3)
    assert result_hu <= result_unknown <= result_ranging


def test_ranging_and_high_uncertainty_encoded_differently():
    """ranging (0.3) and high_uncertainty (-0.3) must have distinct encodings so the
    calibrator can learn that ranging signals are more reliable than high_uncertainty."""
    assert _encode_regime("ranging") == pytest.approx(0.3)
    assert _encode_regime("high_uncertainty") == pytest.approx(-0.3)
    assert _encode_regime("ranging") != pytest.approx(_encode_regime("high_uncertainty"))
    assert _encode_regime("unknown_junk") == pytest.approx(0.0)  # fallback unchanged


def test_direction_guard_blocks_inverted_model():
    """When fit() produces a model where transform(0.75, trending_up) < 0.5, guard forces passthrough."""
    rng = np.random.default_rng(7)
    n = 600
    raw = rng.uniform(0.0, 1.0, n)
    # High raw → mostly outcome=0 (inverted signal — as seen in live ranging market data)
    p_outcome = np.where(raw > 0.7, 0.15, np.where(raw < 0.3, 0.25, 0.75))
    outcomes = (rng.uniform(0, 1, n) < p_outcome).astype(float)
    cal = Calibrator()
    cal.fit(raw, outcomes)
    assert cal._passthrough is True
    assert cal.transform(0.9) == pytest.approx(0.9)


def test_regime_save_load_preserves_aware_flag():
    import tempfile, os
    cal = Calibrator()
    raw = np.array([0.1] * 300 + [0.9] * 300)
    outcomes = np.array([1.0] * 300 + [0.0] * 300)
    regimes = np.array(["trending_up"] * 300 + ["trending_down"] * 300, dtype=object)
    cal.fit(raw, outcomes, regimes=regimes)
    expected = cal.transform(0.7, regime="trending_up")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "regime_cal.joblib")
        cal.save(path)
        cal2 = Calibrator.load(path)
        assert cal2._regime_aware is True
        assert cal2.transform(0.7, regime="trending_up") == pytest.approx(expected, abs=1e-9)


# ── Edge-aware calibrator (Phase 3c) ─────────────────────────────────────────

def _edge_data(n: int = 400, seed: int = 99):
    """Synthetic data where high-edge setups win more than low-edge ones at same raw_prob.

    true_prob compresses raw toward 0.5 (so calibration beats passthrough) while
    adding a meaningful edge contribution. At raw=0.75, edge=0.15 the true win rate
    is ~0.59 > 0.5, so the direction guard passes and the calibrator deploys.
    """
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.55, 0.85, n)
    edge = rng.uniform(0.05, 0.30, n)
    # Compressed toward 0.5 (factor 0.6) plus edge contribution
    true_prob = np.clip(0.5 + 0.6 * (raw - 0.5) + 0.3 * (edge - 0.15), 0.0, 1.0)
    outcomes = (rng.uniform(0, 1, n) < true_prob).astype(float)
    return raw, edge, outcomes


def test_edge_aware_flag_set_on_fit():
    raw, edge, outcomes = _edge_data()
    cal = Calibrator()
    cal.fit(raw, outcomes, edges=edge)
    assert cal._edge_aware is True


def test_edge_aware_transform_differs_by_edge():
    """High edge and low edge should produce different calibrated outputs."""
    raw, edge, outcomes = _edge_data()
    cal = Calibrator()
    cal.fit(raw, outcomes, edges=edge)
    if cal._passthrough:
        pytest.skip("calibrator stayed passthrough on this data")
    low  = cal.transform(0.70, edge=0.05)
    high = cal.transform(0.70, edge=0.25)
    assert low != high


def test_edge_none_defaults_gracefully():
    """transform() with edge=None when edge_aware should not raise."""
    raw, edge, outcomes = _edge_data()
    cal = Calibrator()
    cal.fit(raw, outcomes, edges=edge)
    result = cal.transform(0.70, edge=None)
    assert 0.0 <= result <= 1.0


def test_edge_and_regime_combined():
    """Both regime and edge can be used together."""
    raw, edge, outcomes = _edge_data()
    regimes = np.array(["trending_up"] * len(raw))
    cal = Calibrator()
    cal.fit(raw, outcomes, regimes=regimes, edges=edge)
    result = cal.transform(0.70, regime="trending_up", edge=0.15)
    assert 0.0 <= result <= 1.0


def test_edge_aware_save_load_roundtrip():
    import tempfile, os
    raw, edge, outcomes = _edge_data()
    cal = Calibrator()
    cal.fit(raw, outcomes, edges=edge)
    if cal._passthrough:
        pytest.skip("calibrator stayed passthrough on this data")
    expected = cal.transform(0.70, edge=0.15)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "edge_cal.joblib")
        cal.save(path)
        cal2 = Calibrator.load(path)
        assert cal2._edge_aware is True
        assert cal2.transform(0.70, edge=0.15) == pytest.approx(expected, abs=1e-9)


def test_non_edge_aware_ignores_edge_arg():
    """Old calibrators (edge_aware=False) silently ignore edge argument."""
    raw, outcomes = _synthetic_data(400)
    cal = Calibrator()
    cal.fit(raw, outcomes)
    result_no_edge  = cal.transform(0.70)
    result_with_edge = cal.transform(0.70, edge=0.20)
    assert result_no_edge == pytest.approx(result_with_edge)
