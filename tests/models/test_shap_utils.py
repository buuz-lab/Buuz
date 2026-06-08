from __future__ import annotations

import json

import numpy as np
import pytest
import xgboost as xgb


def _tiny_clf(n_features: int = 3) -> xgb.XGBClassifier:
    """Minimal XGBClassifier for testing SHAP utilities."""
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, (20, n_features))
    y = (X[:, 0] > 0.5).astype(int)
    clf = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42, eval_metric="logloss")
    clf.fit(X, y)
    return clf


# ── compute_coherence ────────────────────────────────────────────────────────

def test_compute_coherence_returns_float():
    from btc_kalshi_system.models.shap_utils import compute_coherence
    clf = _tiny_clf()
    X = np.array([[0.9, 0.8, 0.7]])
    result = compute_coherence(clf, X)
    assert isinstance(result, float)


def test_compute_coherence_in_unit_interval():
    from btc_kalshi_system.models.shap_utils import compute_coherence
    clf = _tiny_clf()
    for row in [[0.9, 0.8, 0.7], [0.1, 0.2, 0.1], [0.5, 0.5, 0.5]]:
        result = compute_coherence(clf, np.array([row]))
        assert 0.0 <= result <= 1.0, f"coherence={result} out of range for {row}"


def test_compute_coherence_matches_manual_calculation():
    """Coherence must equal weighted fraction: sum(|agreeing contribs|) / sum(|all contribs|)."""
    from btc_kalshi_system.models.shap_utils import compute_coherence
    clf = _tiny_clf()
    X = np.array([[0.9, 0.8, 0.7]])
    booster = clf.get_booster()
    contribs = booster.predict(xgb.DMatrix(X), pred_contribs=True)[0]
    feature_contribs = contribs[:-1]
    total = contribs.sum()
    if total == 0 or not np.isfinite(total):
        expected = 0.5
    else:
        pred_sign = 1 if total > 0 else -1
        abs_contribs = np.abs(feature_contribs)
        total_abs = float(abs_contribs.sum())
        if total_abs == 0.0:
            expected = 0.5
        else:
            agree_weight = float(abs_contribs[feature_contribs * pred_sign > 0].sum())
            expected = round(agree_weight / total_abs, 4)
    assert compute_coherence(clf, X) == expected


def test_compute_coherence_zero_prediction_returns_half():
    """When total contribution is exactly zero, coherence defaults to 0.5."""
    from btc_kalshi_system.models.shap_utils import compute_coherence
    clf = _tiny_clf()
    import unittest.mock as mock
    # sum = 0 → total_prediction == 0.0 path returns 0.5
    fake_contribs = np.array([[0.1, -0.1, 0.0, 0.0]])  # sum = 0
    with mock.patch.object(clf.get_booster(), "predict", return_value=fake_contribs):
        result = compute_coherence(clf, np.array([[0.5, 0.5, 0.5]]))
    assert result == 0.5


# ── compute_baseline_snapshot ────────────────────────────────────────────────

def test_compute_baseline_snapshot_structure():
    from btc_kalshi_system.models.shap_utils import compute_baseline_snapshot
    clf = _tiny_clf()
    X = np.array([[0.9, 0.8, 0.7], [0.1, 0.2, 0.1]], dtype=float)
    result = compute_baseline_snapshot(clf, X, ["feat_a", "feat_b", "feat_c"])
    assert set(result.keys()) == {"features", "computed_at", "n_rows"}
    assert result["n_rows"] == 2
    assert len(result["features"]) == 3
    assert all("name" in f and "mean_abs_shap" in f and "importance" in f
               for f in result["features"])


def test_compute_baseline_snapshot_json_serializable():
    from btc_kalshi_system.models.shap_utils import compute_baseline_snapshot
    clf = _tiny_clf()
    X = np.array([[0.9, 0.8, 0.7], [0.1, 0.2, 0.1]], dtype=float)
    result = compute_baseline_snapshot(clf, X, ["feat_a", "feat_b", "feat_c"])
    json.dumps(result)  # must not raise


def test_compute_baseline_snapshot_sorted_descending():
    from btc_kalshi_system.models.shap_utils import compute_baseline_snapshot
    clf = _tiny_clf()
    X = np.random.default_rng(0).uniform(0, 1, (10, 3))
    result = compute_baseline_snapshot(clf, X, ["feat_a", "feat_b", "feat_c"])
    shap_vals = [f["mean_abs_shap"] for f in result["features"]]
    assert shap_vals == sorted(shap_vals, reverse=True)


def test_compute_baseline_snapshot_feature_names_preserved():
    from btc_kalshi_system.models.shap_utils import compute_baseline_snapshot
    clf = _tiny_clf()
    X = np.random.default_rng(0).uniform(0, 1, (10, 3))
    names = ["alpha", "beta", "gamma"]
    result = compute_baseline_snapshot(clf, X, names)
    result_names = {f["name"] for f in result["features"]}
    assert result_names == set(names)


def test_compute_baseline_snapshot_raises_on_length_mismatch():
    from btc_kalshi_system.models.shap_utils import compute_baseline_snapshot
    clf = _tiny_clf()
    X = np.array([[0.9, 0.8, 0.7], [0.1, 0.2, 0.1]], dtype=float)
    with pytest.raises(ValueError, match="feature_names length"):
        compute_baseline_snapshot(clf, X, ["only_two_names"])
