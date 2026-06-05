import os
import tempfile

import numpy as np
import pytest

from btc_kalshi_system.models.regime_model import NotTrainedError, RegimeModel


def _synthetic_features(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 29))  # 29 features: 27 market + kronos_raw_15min + kronos_raw_5min
    y = (X[:, 0] > 0).astype(int)  # label = sign of first feature
    return X, y


def _feature_dict(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "funding_rate":            float(rng.uniform(-0.01, 0.01)),
        "funding_rate_trend":      float(rng.uniform(-0.005, 0.005)),
        "oi_delta_pct":            float(rng.uniform(-0.1, 0.1)),
        "cvd_normalized":          float(rng.uniform(-1, 1)),
        "basis_spread_pct":        float(rng.uniform(-0.01, 0.01)),
        "brti_volatility_1h":      float(rng.uniform(0, 0.02)),
        "cvd_velocity":            float(rng.uniform(-0.1, 0.1)),
        "cvd_acceleration":        float(rng.uniform(-0.05, 0.05)),
        "brti_momentum_5min":      float(rng.uniform(-0.005, 0.005)),
        "brti_momentum_15min":     float(rng.uniform(-0.01, 0.01)),
        "candle_progress":         float(rng.uniform(0, 1)),
        "hour_sin":                float(rng.uniform(-1, 1)),
        "hour_cos":                float(rng.uniform(-1, 1)),
        "funding_window_proximity": float(rng.uniform(0, 1)),
        "trend_slope_1h":          float(rng.uniform(-0.001, 0.001)),
        "trend_r2_1h":             float(rng.uniform(0, 1)),
        "hourly_sr_proximity":     float(rng.uniform(0, 1)),
        "range_breakout_flag":     float(rng.uniform(-1, 1)),
        "tape_speed_tpm":          float(rng.uniform(0, 5)),
        "large_print_direction":   float(rng.uniform(-1, 1)),
        "volume_ratio_1h":         float(rng.uniform(0, 5)),
        "atm_iv":                  float(rng.uniform(20, 100)),
        "iv_rv_spread":            float(rng.uniform(-20, 30)),
        "pcr_oi":                  float(rng.uniform(0.5, 2.0)),
        "term_structure_slope":    float(rng.uniform(-0.3, 0.3)),
        "skew_25d":                float(rng.uniform(-10, 5)),
        "btc_24h_return":          float(rng.uniform(-0.3, 0.3)),
        "kronos_raw_15min":        float(rng.uniform(0.2, 0.8)),
        "kronos_raw_5min":         float(rng.uniform(0.2, 0.8)),
    }


# ── NotTrainedError ────────────────────────────────────────────────────────────

def test_get_regime_raises_not_trained_error_before_training():
    model = RegimeModel()
    with pytest.raises(NotTrainedError):
        model.get_regime(_feature_dict())


def test_not_trained_error_is_runtime_error_subclass():
    assert issubclass(NotTrainedError, RuntimeError)


# ── train / get_regime ─────────────────────────────────────────────────────────

def test_get_regime_returns_dict_after_training():
    model = RegimeModel()
    X, y = _synthetic_features()
    model.train(X, y)
    result = model.get_regime(_feature_dict())
    assert isinstance(result, dict)


def test_get_regime_has_required_keys():
    model = RegimeModel()
    X, y = _synthetic_features()
    model.train(X, y)
    result = model.get_regime(_feature_dict())
    assert "prob_up" in result
    assert "direction" in result
    assert "confidence" in result


def test_get_regime_prob_up_is_float_in_unit_interval():
    model = RegimeModel()
    X, y = _synthetic_features()
    model.train(X, y)
    result = model.get_regime(_feature_dict())
    assert isinstance(result["prob_up"], float)
    assert 0.0 <= result["prob_up"] <= 1.0


def test_get_regime_direction_is_zero_or_one():
    model = RegimeModel()
    X, y = _synthetic_features()
    model.train(X, y)
    result = model.get_regime(_feature_dict())
    assert result["direction"] in (0, 1)


def test_get_regime_confidence_is_float_in_unit_interval():
    model = RegimeModel()
    X, y = _synthetic_features()
    model.train(X, y)
    result = model.get_regime(_feature_dict())
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


def test_direction_consistent_with_prob_up():
    """direction=1 iff prob_up >= 0.5."""
    model = RegimeModel()
    X, y = _synthetic_features()
    model.train(X, y)
    for seed in range(10):
        result = model.get_regime(_feature_dict(seed))
        if result["prob_up"] >= 0.5:
            assert result["direction"] == 1
        else:
            assert result["direction"] == 0


# ── save / load ────────────────────────────────────────────────────────────────

def test_save_and_load_roundtrip_preserves_predictions():
    model = RegimeModel()
    X, y = _synthetic_features()
    model.train(X, y)
    features = _feature_dict()
    expected = model.get_regime(features)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "regime.joblib")
        model.save(path)
        model2 = RegimeModel.load(path)
        result = model2.get_regime(features)
        assert result["prob_up"] == pytest.approx(expected["prob_up"], abs=1e-9)
        assert result["direction"] == expected["direction"]


def test_load_from_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        RegimeModel.load("/tmp/does_not_exist_regime.joblib")


def test_loaded_model_raises_not_trained_error_when_file_missing():
    """RegimeModel() with no prior training raises NotTrainedError, not a crash."""
    model = RegimeModel()
    with pytest.raises(NotTrainedError):
        model.get_regime(_feature_dict())


# ── Phase 2: Kronos meta-features ─────────────────────────────────────────────

def test_feature_order_includes_kronos_features():
    """_FEATURE_ORDER must include kronos_raw_15min and kronos_raw_5min as regime meta-features."""
    from btc_kalshi_system.models.regime_model import _FEATURE_ORDER
    assert "kronos_raw_15min" in _FEATURE_ORDER
    assert "kronos_raw_5min" in _FEATURE_ORDER


def test_get_regime_handles_none_kronos_features():
    """get_regime() must not crash when kronos_raw_15min / kronos_raw_5min are None (bootstrap)."""
    model = RegimeModel()
    X, y = _synthetic_features()
    model.train(X, y)
    fd = _feature_dict()
    fd["kronos_raw_15min"] = None
    fd["kronos_raw_5min"] = None
    result = model.get_regime(fd)
    assert isinstance(result["prob_up"], float)
