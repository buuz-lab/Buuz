# Calibration Curve + SHAP Coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SHAP-based directional coherence scoring to the regime model pipeline and replace the rigid contextuality check with a real calibration curve in the monitor and dry-run output.

**Architecture:** `shap_utils.py` provides two functions used by three consumers: `regime_model.get_regime()` computes per-prediction coherence and adds it to its return dict; `train_regime.py` saves a training-set SHAP baseline snapshot after each train; `regime_v2_monitor.py` reads both for Section 5. Coherence flows through `TradingSignal` → `gate_rejections` so `train_calibrator.py` can use it as an 8th calibrator input using the identical pattern as `brti_volatility_1h` and `kalshi_spread_normalized`.

**Tech Stack:** Python 3.11, XGBoost (`pred_contribs=True`), SQLite, pytest

---

## File Map

| File | Change |
|---|---|
| `btc_kalshi_system/models/shap_utils.py` | **NEW** — `compute_coherence`, `compute_baseline_snapshot` |
| `tests/models/test_shap_utils.py` | **NEW** — unit tests for both functions |
| `btc_kalshi_system/models/regime_model.py` | `get_regime()` adds `shap_coherence` to return dict |
| `btc_kalshi_system/signal/fusion.py` | `TradingSignal` gets `shap_coherence` field; `get_signal()` sets it; `get_features_snapshot()` returns it |
| `main.py` | Schema migrations for `candle_features` + `gate_rejections`; unpack 5-tuple from `get_features_snapshot()`; write `shap_coherence` in both INSERTs |
| `scripts/train_regime.py` | Save `regime_shap_baseline.json` after train; replace contextuality check with calibration sanity |
| `scripts/regime_v2_monitor.py` | Section 5: calibration curve + SHAP feature contributions |
| `btc_kalshi_system/models/calibrator.py` | Add `shap_coherences` as 8th optional input to `_build_X`, `fit`, `transform`, `save`, `load` |
| `scripts/train_calibrator.py` | Add `shap_coherence` to UNION query and `Calibrator.fit()` call |
| `tests/models/test_regime_model.py` | Update `test_get_regime_has_required_keys` |
| `tests/models/test_calibrator.py` | Add tests for `shap_coherence_aware` flag |

---

## Task 1: `shap_utils.py` — new module + tests

**Files:**
- Create: `btc_kalshi_system/models/shap_utils.py`
- Create: `tests/models/test_shap_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/models/test_shap_utils.py`:

```python
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
    """Coherence must equal the fraction computed directly from pred_contribs."""
    from btc_kalshi_system.models.shap_utils import compute_coherence
    clf = _tiny_clf()
    X = np.array([[0.9, 0.8, 0.7]])
    contribs = clf.predict(X, pred_contribs=True)[0]
    feature_contribs = contribs[:-1]
    total = contribs.sum()
    if total == 0:
        expected = 0.5
    else:
        pred_sign = 1 if total > 0 else -1
        expected = round(float(np.sum(feature_contribs * pred_sign > 0) / len(feature_contribs)), 4)
    assert compute_coherence(clf, X) == expected


def test_compute_coherence_zero_prediction_returns_half():
    """When total contribution is exactly zero, coherence defaults to 0.5."""
    from btc_kalshi_system.models.shap_utils import compute_coherence
    clf = _tiny_clf()
    # Use a balanced row — exact 0 is hard to hit in practice but we mock contribs
    import unittest.mock as mock
    fake_contribs = np.array([[0.1, -0.1, 0.0, 0.0]])  # sum = 0 (bias+features)
    with mock.patch.object(clf, "predict", return_value=fake_contribs):
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/models/test_shap_utils.py -v 2>&1 | tail -15
```

Expected: `ModuleNotFoundError: No module named 'btc_kalshi_system.models.shap_utils'`

- [ ] **Step 3: Create `btc_kalshi_system/models/shap_utils.py`**

```python
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np


def compute_coherence(clf, X: np.ndarray) -> float:
    """
    Fraction of the model's feature SHAP contributions pointing in the same
    direction as the final prediction. Score near 1.0 = coherent feature
    agreement; near 0.5 = features split, one dominant feature driving the call.

    Uses XGBoost's built-in pred_contribs — no external shap library needed.
    X must be shape (1, n_features).
    """
    contribs = clf.predict(X, pred_contribs=True)[0]   # shape: (n_features + 1,)
    feature_contribs = contribs[:-1]                    # drop bias column
    total_prediction = contribs.sum()                   # full margin including bias
    if total_prediction == 0.0:
        return 0.5
    pred_sign = 1 if total_prediction > 0 else -1
    n_agree = int(np.sum(feature_contribs * pred_sign > 0))
    return round(float(n_agree / len(feature_contribs)), 4)


def compute_baseline_snapshot(
    clf,
    X_train: np.ndarray,
    feature_names: list[str],
) -> dict:
    """
    Mean absolute SHAP per feature over the training set, sorted descending.
    Returns a JSON-serializable dict for saving to models/regime_shap_baseline.json.
    Updated on every train_regime.py run.
    """
    contribs = clf.predict(X_train, pred_contribs=True)[:, :-1]  # drop bias
    mean_abs_shap = np.nanmean(np.abs(contribs), axis=0)
    importances = clf.feature_importances_
    total_imp = float(importances.sum()) or 1.0

    ranked = sorted(
        zip(feature_names, mean_abs_shap.tolist(), importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return {
        "features": [
            {
                "name": name,
                "mean_abs_shap": round(shap, 6),
                "importance": round(imp / total_imp, 6),
            }
            for name, shap, imp in ranked
        ],
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(X_train)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/models/test_shap_utils.py -v 2>&1 | tail -15
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/models/shap_utils.py tests/models/test_shap_utils.py && git commit -m "$(cat <<'EOF'
feat: add shap_utils — compute_coherence + compute_baseline_snapshot

Directional SHAP coherence: fraction of 41 features pointing same direction
as prediction. Replaces rigid contextuality check with a data-driven measure.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `regime_model.get_regime()` — add `shap_coherence` to return dict

**Files:**
- Modify: `btc_kalshi_system/models/regime_model.py`
- Modify: `tests/models/test_regime_model.py`

- [ ] **Step 1: Update the failing test**

In `tests/models/test_regime_model.py`, find `test_get_regime_has_required_keys` and update:

```python
def test_get_regime_has_required_keys():
    model = RegimeModel()
    X = np.array([[v if v is not None else float("nan")
                   for v in _feature_dict().values()]])
    y = np.array([1])
    model.train(X, y)
    result = model.get_regime(_feature_dict())
    assert set(result.keys()) == {"prob_up", "direction", "confidence", "shap_coherence"}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/models/test_regime_model.py::test_get_regime_has_required_keys -v 2>&1 | tail -10
```

Expected: FAIL — `assert {'prob_up', 'direction', 'confidence'} == {'prob_up', 'direction', 'confidence', 'shap_coherence'}`

- [ ] **Step 3: Update `get_regime()` in `btc_kalshi_system/models/regime_model.py`**

Read `btc_kalshi_system/models/regime_model.py` lines 88–99. Replace the `get_regime` method body:

```python
    def get_regime(self, features: dict) -> dict:
        if self._clf is None:
            raise NotTrainedError(
                "RegimeModel has not been trained. Call train() or load() first."
            )
        from btc_kalshi_system.models.shap_utils import compute_coherence
        X = np.array([[features[k] if features[k] is not None else float("nan") for k in _FEATURE_ORDER]])
        prob_up = float(self._clf.predict_proba(X)[0, 1])
        direction = int(prob_up >= 0.5)
        confidence = float(abs(prob_up - 0.5) * 2)  # 0 at boundary, 1 at extremes
        shap_coherence = compute_coherence(self._clf, X)
        return {"prob_up": prob_up, "direction": direction, "confidence": confidence, "shap_coherence": shap_coherence}
```

- [ ] **Step 4: Run all regime model tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/models/test_regime_model.py -v 2>&1 | tail -20
```

Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/models/regime_model.py tests/models/test_regime_model.py && git commit -m "$(cat <<'EOF'
feat: regime_model.get_regime() returns shap_coherence

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `fusion.py` + `main.py` — propagate `shap_coherence` end-to-end

**Files:**
- Modify: `btc_kalshi_system/signal/fusion.py`
- Modify: `main.py`

- [ ] **Step 1: Add `shap_coherence` field to `TradingSignal` in `fusion.py`**

Read `btc_kalshi_system/signal/fusion.py` lines 63–92 to find the `TradingSignal` dataclass. Add `shap_coherence` after the `okx_stale` field:

```python
    okx_stale: bool = False
    # SHAP directional coherence at signal time — fraction of features pointing
    # same direction as prediction. None when model not trained (bootstrap mode).
    shap_coherence: float | None = None
```

- [ ] **Step 2: Set `shap_coherence` in `get_signal()` in `fusion.py`**

Read `fusion.py` around line 185 where `regime_features, features_stale, ...` is unpacked. Just before the `try:` block that calls `self._regime.get_regime(...)`, add:

```python
        regime_shap_coherence: float | None = None
```

Then inside the `try` block, after `regime_prob = regime_result["prob_up"]`, add:

```python
            regime_shap_coherence = regime_result.get("shap_coherence")
```

Then find the `return TradingSignal(...)` call (around line 249, after the try/except) and add before the closing paren:

```python
            shap_coherence=regime_shap_coherence,
```

In the `except NotTrainedError` block (bootstrap path), `regime_shap_coherence` stays `None` from initialization — no change needed.

- [ ] **Step 3: Update `get_features_snapshot()` in `fusion.py` to return `shap_coherence`**

Read `fusion.py` lines 262–275. Replace the method:

```python
    def get_features_snapshot(self) -> tuple[dict, bool, bool, float | None, float | None]:
        """
        Returns (features_dict, features_stale, deribit_stale, regime_prob, shap_coherence).
        regime_prob and shap_coherence are None when model not trained or paused.
        """
        features, features_stale, deribit_stale, _okx_stale = self._regime_features()
        regime_prob: float | None = None
        shap_coherence: float | None = None
        if not _REGIME_PAUSE_FLAG.exists():
            try:
                regime_result = self._regime.get_regime(features)
                regime_prob = regime_result["prob_up"]
                shap_coherence = regime_result["shap_coherence"]
            except NotTrainedError:
                pass
        return features, features_stale, deribit_stale, regime_prob, shap_coherence
```

- [ ] **Step 4: Add schema migrations in `main.py`**

Read `main.py` around line 325. In `_CANDLE_FEATURES_COLUMN_MIGRATIONS`, add after `("k15_delta", "REAL DEFAULT NULL")`:

```python
    ("shap_coherence",            "REAL DEFAULT NULL"),
```

Read `main.py` around line 206. In `_GATE_REJECTIONS_COLUMN_MIGRATIONS`, add after `("kalshi_spread_normalized", "REAL DEFAULT NULL")`:

```python
    ("shap_coherence",          "REAL DEFAULT NULL"),
```

- [ ] **Step 5: Update `get_features_snapshot()` call site in `main.py`**

In `main.py` around line 867, find:

```python
                features, features_stale, deribit_stale, regime_prob = self._fusion.get_features_snapshot()
```

Replace with:

```python
                features, features_stale, deribit_stale, regime_prob, shap_coherence = self._fusion.get_features_snapshot()
```

- [ ] **Step 6: Add `shap_coherence` to the `candle_features` INSERT in `main.py`**

Find the col_names string (around line 937) and change:

```python
                    "spread_change, oi_delta_at_midcandle, k5_candle_ts, mid_candle_model_prob, "
                    + ", ".join(cols)
```

To:

```python
                    "spread_change, oi_delta_at_midcandle, k5_candle_ts, mid_candle_model_prob, "
                    "shap_coherence, "
                    + ", ".join(cols)
```

Find `mid_candle_model_prob,` in the values list (around line 979) and add after it:

```python
                        shap_coherence,
```

Update placeholder count from `36` to `37`:

```python
                placeholders = ", ".join(["?"] * (37 + len(cols)))
```

- [ ] **Step 7: Add `shap_coherence` to the `gate_rejections` INSERT in `main.py`**

Find the gate_rejections INSERT (around line 1336). Add `shap_coherence` to the column list:

```python
                    """INSERT OR IGNORE INTO gate_rejections
                       (rejection_id, timestamp, ticker, timeframe, direction,
                        failed_gate, failed_reason, signal_prob, deepseek_regime,
                        kalshi_mid_cents, features, kalshi_mid_at_block, would_be_fill_cents,
                        kronos_raw_15min, kronos_raw, k15_calibrated_prob, candle_progress,
                        k15_post_open, regime_prob, signal_edge,
                        brti_volatility_1h, kalshi_spread_normalized, shap_coherence)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
```

And add `shap_coherence` value at the end of the values tuple:

```python
                        _rf.get("kalshi_spread_normalized"),
                        signal.shap_coherence,
```

- [ ] **Step 8: Verify `main.py` parses**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -c "import main; print('OK')"
```

Expected: `OK`

- [ ] **Step 9: Run tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/signal/ tests/models/ --tb=short -q 2>&1 | tail -10
```

Expected: All passing, no failures.

- [ ] **Step 10: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/signal/fusion.py main.py && git commit -m "$(cat <<'EOF'
feat: propagate shap_coherence through TradingSignal to candle_features + gate_rejections

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `train_regime.py` — baseline snapshot + calibration sanity check

**Files:**
- Modify: `scripts/train_regime.py`

- [ ] **Step 1: Read the existing contextuality block**

Read `scripts/train_regime.py` lines 285–366 to see the full contextuality check block that will be replaced.

- [ ] **Step 2: Replace the contextuality check with calibration sanity**

Find and replace the entire contextuality block (from `# ── Kronos contextuality check ──` through the final `print("──────────────────────────────────────────────────────────────────────")` after the try/except):

```python
    # ── Calibration sanity check ──────────────────────────────────────────────
    # Replaces the rigid contextuality probe (k15=0.85 in ranging vs trending).
    # Checks whether model confidence correlates with actual accuracy on the
    # training set — the real signal of whether the model learned something useful.
    print("\n── Calibration sanity (training predictions) ─────────────────────────")
    try:
        train_proba = model._clf.predict_proba(X_train)[:, 1]
        tiers = [
            ("Low  (|p-0.5|<0.10)", lambda p: abs(p - 0.5) < 0.10),
            ("Med  (0.10–0.20)",     lambda p: 0.10 <= abs(p - 0.5) < 0.20),
            ("High (|p-0.5|>0.20)", lambda p: abs(p - 0.5) >= 0.20),
        ]
        tier_accs = {}
        for tier_name, tier_fn in tiers:
            mask = np.array([tier_fn(p) for p in train_proba])
            n = int(mask.sum())
            if n == 0:
                print(f"  {tier_name:<22s}  n=0   (no predictions in this range)")
                continue
            tier_y = y_train[mask]
            tier_p = train_proba[mask]
            brier  = float(np.mean((tier_p - tier_y) ** 2))
            acc    = float(((tier_p >= 0.5).astype(int) == tier_y).mean())
            tier_accs[tier_name] = acc
            if n >= 10:
                win_rate = float(tier_y.mean()) * 100
                print(f"  {tier_name:<22s}  n={n:<4d}  win_rate={win_rate:.0f}%  Brier={brier:.3f}  acc={acc:.0%}")
            else:
                print(f"  {tier_name:<22s}  n={n:<4d}  (accumulating — need 10+ for stats)")

        # Gradient check: high-confidence predictions should be more accurate
        low_key  = "Low  (|p-0.5|<0.10)"
        high_key = "High (|p-0.5|>0.20)"
        if low_key in tier_accs and high_key in tier_accs:
            if tier_accs[high_key] > tier_accs[low_key]:
                print(f"  ✓ Calibration gradient present — high acc={tier_accs[high_key]:.0%} > low acc={tier_accs[low_key]:.0%}")
            else:
                print(f"  ⚠ No calibration gradient — high acc={tier_accs[high_key]:.0%} ≤ low acc={tier_accs[low_key]:.0%}")

        k15_imp = importances[_FEATURE_COLS.index("kronos_raw_15min")] / total_imp * 100 if total_imp > 0 else 0
        k5_imp  = importances[_FEATURE_COLS.index("kronos_raw_5min")]  / total_imp * 100 if total_imp > 0 else 0
        print(f"  Kronos importance: k15={k15_imp:.1f}%  k5={k5_imp:.1f}%  combined={k15_imp+k5_imp:.1f}%")

    except Exception as exc:
        print(f"  Could not run calibration sanity check: {exc}")
    print("──────────────────────────────────────────────────────────────────────")
```

- [ ] **Step 3: Add baseline snapshot save after model.save()**

Read `scripts/train_regime.py` around the `model.save(args.out)` call (after the `if not args.dry_run:` block). Add immediately after `model.save(args.out)`:

```python
        # Save SHAP baseline snapshot alongside model for monitor + future analysis.
        try:
            import json
            from pathlib import Path
            from btc_kalshi_system.models.shap_utils import compute_baseline_snapshot
            snapshot = compute_baseline_snapshot(model._clf, X_train, _FEATURE_COLS)
            shap_path = Path(args.out).parent / "regime_shap_baseline.json"
            shap_path.write_text(json.dumps(snapshot, indent=2))
            print(f"  SHAP baseline saved → {shap_path} (n={snapshot['n_rows']} rows, {len(snapshot['features'])} features)")
        except Exception as exc:
            print(f"  SHAP baseline save failed (non-fatal): {exc}")
```

- [ ] **Step 4: Run dry-run to verify new output format**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 scripts/train_regime.py --dry-run --min-rows 600 2>&1 | grep -A 20 "Calibration sanity"
```

Expected: Shows three tiers with Brier and win_rate. No `contextuality` mention. No crash.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add scripts/train_regime.py && git commit -m "$(cat <<'EOF'
feat: replace contextuality check with calibration sanity in train_regime.py

Removes synthetic k15=0.85 probe. Adds real calibration gradient check:
does high-confidence tier have better accuracy than low-confidence?
Saves regime_shap_baseline.json after every train.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `regime_v2_monitor.py` — Section 5

**Files:**
- Modify: `scripts/regime_v2_monitor.py`

- [ ] **Step 1: Add the import and constant**

Read `scripts/regime_v2_monitor.py` lines 1–40 (imports + constants). Add at the top of the constants section after the existing constants:

```python
_SHAP_BASELINE_PATH = "models/regime_shap_baseline.json"
```

Also verify `import json` and `from pathlib import Path` are present; add them if not.

- [ ] **Step 2: Add `section_calibration_and_shap()` function**

Add this new function after `section_training_health()` and before `main()`:

```python
def section_calibration_and_shap(conn: sqlite3.Connection) -> list[str]:
    """Section 5: calibration curve + SHAP feature contributions."""
    issues: list[str] = []
    print(f"\n── Calibration Curve + SHAP {'─'*43}")

    # ── 5a: Calibration curve by regime_prob confidence tier ─────────────────
    rows = conn.execute("""
        SELECT regime_prob, btc_direction, kalshi_open_mid, shap_coherence
        FROM candle_features
        WHERE features_stale=0 AND regime_prob IS NOT NULL
          AND btc_direction IS NOT NULL AND kalshi_open_mid IS NOT NULL
        ORDER BY candle_ts ASC
    """).fetchall()

    n_total = len(rows)
    if n_total == 0:
        print("  (no regime_prob rows yet — check back after regime v2 deploys tonight)")
    else:
        tiers = [
            ("Low  (|p-0.5|<0.10)", lambda p: abs(p - 0.5) < 0.10,  False),
            ("Med  (0.10–0.20)",     lambda p: 0.10 <= abs(p - 0.5) < 0.20, False),
            ("High (|p-0.5|>0.20)", lambda p: abs(p - 0.5) >= 0.20, True),
        ]
        high_regime_brier = high_kalshi_brier = high_n = None

        for tier_name, tier_fn, is_high in tiers:
            tier_rows = [(p, y, k, c) for p, y, k, c in rows if tier_fn(p)]
            n = len(tier_rows)
            if n == 0:
                print(f"  {tier_name:<24s}  n=0   (accumulating)")
                continue
            regime_ps  = [r[0] for r in tier_rows]
            ys         = [r[1] for r in tier_rows]
            kalshi_ks  = [r[2] for r in tier_rows]
            coherences = [r[3] for r in tier_rows if r[3] is not None]

            r_brier = sum((p - y) ** 2 for p, y in zip(regime_ps, ys)) / n
            k_brier = sum((k - y) ** 2 for k, y in zip(kalshi_ks, ys)) / n
            adv = (k_brier - r_brier) / k_brier * 100 if k_brier > 0 else 0

            if n >= 10:
                win_rate = sum(ys) / n * 100
                coh_str = f"  coh={sum(coherences)/len(coherences):.2f}" if coherences else ""
                print(f"  {tier_name:<24s}  n={n:<4d}  win={win_rate:.0f}%  "
                      f"regime={r_brier:.3f}  kalshi={k_brier:.3f}  adv={adv:+.1f}%{coh_str}")
            else:
                print(f"  {tier_name:<24s}  n={n:<4d}  (accumulating — need 10+)")

            if is_high:
                high_regime_brier, high_kalshi_brier, high_n = r_brier, k_brier, n

        # Key signal: does model beat Kalshi in the high-confidence tier?
        if high_n is not None and high_n >= 10:
            if high_regime_brier < high_kalshi_brier:
                print(f"  {_STATUS['PASS']} HIGH tier beats Kalshi — go-live signal present")
            else:
                msg = f"High-confidence regime ({high_regime_brier:.3f}) ≥ Kalshi ({high_kalshi_brier:.3f})"
                print(f"  {_STATUS['WARN']} {msg}")
                issues.append(f"WARN  {msg}")

        print(f"  Total regime_prob rows: {n_total}")

    # ── 5b: SHAP feature contributions ───────────────────────────────────────
    shap_path = Path(_SHAP_BASELINE_PATH)
    if shap_path.exists():
        try:
            snapshot = json.loads(shap_path.read_text())
            n_snap   = snapshot.get("n_rows", 0)
            computed = snapshot.get("computed_at", "unknown")[:19]
            print(f"\n  SHAP baseline  (n_train={n_snap}, updated {computed})")
            print(f"  {'Feature':<28s}  {'Mean|SHAP|':>10s}  {'Importance':>10s}")
            for feat in snapshot["features"][:10]:
                print(f"  {feat['name']:<28s}  {feat['mean_abs_shap']:>10.4f}  {feat['importance']:>10.3%}")
        except Exception as exc:
            print(f"  {_STATUS['WARN']} Could not read SHAP baseline: {exc}")
    else:
        print("\n  (SHAP baseline not yet available — appears after first train_regime.py run)")

    return issues
```

- [ ] **Step 3: Add the section call in `main()`**

Read `scripts/regime_v2_monitor.py` in the `main()` function, find the try block that calls all sections:

```python
        all_issues += section_api_health(conn, args.hours)
        all_issues += section_distribution_drift(conn, args.hours)
        all_issues += section_kalshi_edge(conn)
        all_issues += section_training_health(conn)
```

Add after `section_training_health`:

```python
        all_issues += section_calibration_and_shap(conn)
```

- [ ] **Step 4: Run the monitor to verify it works**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 scripts/regime_v2_monitor.py 2>&1 | grep -A 20 "Calibration Curve"
```

Expected: Shows the section header, then either "(no regime_prob rows yet)" or calibration tiers, and "(SHAP baseline not yet available)". No crash.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add scripts/regime_v2_monitor.py && git commit -m "$(cat <<'EOF'
feat: regime_v2_monitor Section 5 — calibration curve + SHAP feature contributions

Shows Brier by confidence tier (low/med/high) vs Kalshi, SHAP coherence
per tier, and top-10 features by mean absolute SHAP from baseline snapshot.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `calibrator.py` + `train_calibrator.py` — 8th input

**Files:**
- Modify: `btc_kalshi_system/models/calibrator.py`
- Modify: `scripts/train_calibrator.py`
- Modify: `tests/models/test_calibrator.py`

- [ ] **Step 1: Write the new failing test**

Add to `tests/models/test_calibrator.py` after the existing `test_spread_aware_flag_set_on_fit` test:

```python
def test_shap_coherence_aware_flag_set_on_fit():
    cal = Calibrator()
    raw, outcomes = _compressed_data(n=300)
    coherences = np.full(300, 0.75)
    cal.fit(raw, outcomes, shap_coherences=coherences)
    assert cal._shap_coherence_aware is True


def test_shap_coherence_aware_false_without_coherences():
    cal = Calibrator()
    raw, outcomes = _compressed_data(n=300)
    cal.fit(raw, outcomes)
    assert cal._shap_coherence_aware is False


def test_shap_coherence_save_load_preserves_flag():
    import tempfile, os
    cal = Calibrator()
    raw, outcomes = _compressed_data(n=300)
    cal.fit(raw, outcomes, shap_coherences=np.full(300, 0.75))
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name
    try:
        cal.save(path)
        loaded = Calibrator.load(path)
        assert loaded._shap_coherence_aware == cal._shap_coherence_aware
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/models/test_calibrator.py::test_shap_coherence_aware_flag_set_on_fit tests/models/test_calibrator.py::test_shap_coherence_aware_false_without_coherences tests/models/test_calibrator.py::test_shap_coherence_save_load_preserves_flag -v 2>&1 | tail -10
```

Expected: 3 FAIL — `TypeError: fit() got an unexpected keyword argument 'shap_coherences'`

- [ ] **Step 3: Update `Calibrator.__init__()` in `calibrator.py`**

Read `btc_kalshi_system/models/calibrator.py` lines 28–43 (`__init__`). Add after `self._spread_aware = False`:

```python
        self._shap_coherence_aware: bool = False
```

- [ ] **Step 4: Update `Calibrator._build_X()` in `calibrator.py`**

Read `calibrator.py` lines 50–70 (`_build_X`). Add `shap_coherences` parameter and column:

```python
    def _build_X(
        self,
        raw: np.ndarray,
        regime_scores: np.ndarray | None,
        edges: np.ndarray | None,
        disagreements: np.ndarray | None = None,
        volatilities: np.ndarray | None = None,
        spreads: np.ndarray | None = None,
        shap_coherences: np.ndarray | None = None,
    ) -> np.ndarray:
        cols = [raw, raw ** 2]
        if regime_scores is not None:
            cols.append(regime_scores)
        if edges is not None:
            cols.append(edges)
        if disagreements is not None:
            cols.append(disagreements)
        if volatilities is not None:
            cols.append(volatilities)
        if spreads is not None:
            cols.append(spreads)
        if shap_coherences is not None:
            cols.append(shap_coherences)
        return np.column_stack(cols)
```

- [ ] **Step 5: Update `Calibrator.fit()` in `calibrator.py`**

Add `shap_coherences: np.ndarray | None = None` to the `fit()` signature after `spreads`. Then add the same handling pattern as `spreads` throughout the method body:

After `use_spread = ...` add:
```python
        use_shap_coherence = shap_coherences is not None and len(shap_coherences) == n
```

After `spread_arr = ...` add:
```python
        shap_coherence_arr = np.asarray(shap_coherences, dtype=float) if use_shap_coherence else None
```

After `def _split(arr): ...` update the splits:
```python
        spr_tr, spr_ho = _split(spread_arr)
        shc_tr, shc_ho = _split(shap_coherence_arr)
```

Update both `_build_X` calls to pass `shap_coherences`:
```python
        X_train   = self._build_X(raw_train,   reg_tr, edg_tr, dis_tr, vol_tr, spr_tr, shc_tr)
        X_holdout = self._build_X(raw_holdout, reg_ho, edg_ho, dis_ho, vol_ho, spr_ho, shc_ho)
```

Update the direction guard `_build_X` call:
```python
            guard_X = self._build_X(guard_raw, guard_reg, guard_edg, guard_dis, guard_vol, guard_spr, None)
```

After `self._spread_aware = use_spread` add:
```python
                self._shap_coherence_aware = use_shap_coherence
```

- [ ] **Step 6: Update `Calibrator.transform()` in `calibrator.py`**

Add `shap_coherence: float | None = None` to `transform()` signature. Then add after the `spr` line:

```python
        shc = np.array([shap_coherence if shap_coherence is not None else 0.5]) if self._shap_coherence_aware else None
```

Update the `_build_X` call:
```python
        X = self._build_X(raw, reg, edg, dis, vol, spr, shc)
```

- [ ] **Step 7: Update `save()` and `load()` in `calibrator.py`**

In `save()`, add after `"spread_aware": self._spread_aware`:
```python
            "shap_coherence_aware": self._shap_coherence_aware,
```

In `load()`, add after `obj._spread_aware = state.get("spread_aware", False)`:
```python
        obj._shap_coherence_aware = state.get("shap_coherence_aware", False)
```

- [ ] **Step 8: Run calibrator tests to verify they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/models/test_calibrator.py -v 2>&1 | tail -20
```

Expected: All tests pass including the 3 new ones.

- [ ] **Step 9: Update `train_calibrator.py` SQL and fit call**

Read `scripts/train_calibrator.py` lines 55–90. Find `_UNION_QUERY`. Add `shap_coherence` to both SELECT statements in the UNION:

```python
_UNION_QUERY = """
    SELECT regime_prob, signal_edge, deepseek_regime, direction, outcome,
           kronos_raw_15min, brti_volatility_1h, kalshi_spread_normalized,
           shap_coherence FROM (
        SELECT regime_prob, signal_edge, deepseek_regime, direction, outcome, timestamp,
               kronos_raw_15min, brti_volatility_1h, kalshi_spread_normalized,
               shap_coherence
        FROM trades
        WHERE outcome IS NOT NULL AND regime_prob IS NOT NULL AND signal_edge IS NOT NULL
        UNION ALL
        SELECT regime_prob, signal_edge, deepseek_regime, direction, outcome, timestamp,
               kronos_raw_15min, brti_volatility_1h, kalshi_spread_normalized,
               shap_coherence
        FROM gate_rejections
        WHERE outcome IS NOT NULL AND shadow = 0
          AND regime_prob IS NOT NULL AND signal_edge IS NOT NULL
    )
    ORDER BY timestamp DESC LIMIT ?
"""
```

After `spreads = np.nan_to_num(spreads, nan=0.0)` (around line 125), add:

```python
    shap_coherences_raw = np.array([r[8] if r[8] is not None else np.nan for r in rows], dtype=float)
    shap_coherences = None if np.all(np.isnan(shap_coherences_raw)) else np.nan_to_num(shap_coherences_raw, nan=0.5)
```

Find `cal.fit(` (around line 130, where `regime_probs, y_yes, regimes=regimes, ...` are passed) and add `shap_coherences=shap_coherences`:

```python
    cal.fit(
        regime_probs, y_yes,
        regimes=regimes,
        edges=abs_edges,
        disagreements=disagreements,
        volatilities=volatilities,
        spreads=spreads,
        shap_coherences=shap_coherences,
    )
```

- [ ] **Step 10: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: 607+ passing, 0 failures.

- [ ] **Step 11: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/models/calibrator.py scripts/train_calibrator.py tests/models/test_calibrator.py && git commit -m "$(cat <<'EOF'
feat: shap_coherence as 8th calibrator input (Phase 3c foundation)

Follows existing spread_aware/volatility_aware pattern. NULL rows use 0.5
(neutral) default. train_calibrator.py reads from gate_rejections UNION.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Full suite + restart + push

- [ ] **Step 1: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: 607+ passing, 0 failures.

- [ ] **Step 2: Restart service**

```bash
launchctl kickstart -k gui/$(id -u)/com.kronos.v2
```

- [ ] **Step 3: Verify service started and shap_coherence column exists**

```bash
sleep 15 && cd "/Users/ezrakornberg/Kronos V2" && python3 -c "
import sqlite3
conn = sqlite3.connect('trades.db')
cols = [r[1] for r in conn.execute('PRAGMA table_info(candle_features)').fetchall()]
print('shap_coherence in candle_features:', 'shap_coherence' in cols)
gr_cols = [r[1] for r in conn.execute('PRAGMA table_info(gate_rejections)').fetchall()]
print('shap_coherence in gate_rejections:', 'shap_coherence' in gr_cols)
conn.close()
"
```

Expected: both `True`.

- [ ] **Step 4: Push**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git push origin main
```
