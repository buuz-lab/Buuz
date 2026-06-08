# Calibration Curve + SHAP Coherence Design

**Date:** 2026-06-08  
**Status:** Approved  
**Motivation:** The contextuality check (`k15=0.85 in trending vs ranging`) is too rigid. Market regime is a continuous multi-dimensional property, not a four-label category. This design replaces it with two honest measures: a calibration curve that shows whether model confidence correlates with actual accuracy, and a SHAP coherence score that captures whether the model's features agree directionally — the real signal of prediction quality.

---

## The Thesis This Replaces

**Old thesis (contextuality gap):** "The model should respond differently to k15=0.85 when DeepSeek labels the regime as 'trending' vs 'ranging'."

**Why it's wrong:** This bakes in that DeepSeek's four discrete labels are the right axis of variation, and that k15 specifically is the lever worth probing. The model can learn rich multi-dimensional interactions that produce no gap on this synthetic probe while still being excellent.

**New thesis:** The model has edge when its confidence is **calibrated** (high predicted probability = high actual win rate) and **coherent** (multiple independent features agree directionally). These two properties are measurable without reference to any discrete regime label.

---

## What "SHAP Coherence" Means

XGBoost's `predict(pred_contribs=True)` returns the contribution of each of the 41 features to the final prediction for a specific candle. A positive contribution means the feature pushed the prediction toward UP; negative means toward DOWN.

**Coherence score** = fraction of the 41 feature contributions pointing in the same direction as the final prediction.

- Score near 1.0: features broadly agree — the model is finding consistent evidence
- Score near 0.5: features split — the prediction is driven by a few dominant features while others push back

Two candles can both show `regime_prob = 0.72`, but:
- Candle A: coherence = 0.82 → 34 of 41 features point UP. Strong, trustworthy call.
- Candle B: coherence = 0.54 → 22 features point UP, 19 point DOWN. One feature overwhelmed the others.

`signal_edge` (how far regime_prob is from Kalshi) cannot distinguish these. SHAP coherence can.

---

## Components

### 1. `btc_kalshi_system/models/shap_utils.py` (new file)

Single-responsibility module. Two public functions:

**`compute_coherence(clf, feature_vector: np.ndarray) -> float`**
- Calls `clf.predict(feature_vector, pred_contribs=True)`
- Returns: `(# contributions pointing same direction as prediction bias) / 41`
- Direction of prediction bias = `1` if `pred_contribs.sum() > 0`, else `-1`  
- Bias column (last element of pred_contribs output) excluded from the count
- Returns `0.5` if model is not trained or prediction is exactly at base rate
- No external dependencies — uses only XGBoost's built-in prediction API

**`compute_baseline_snapshot(clf, X_train: np.ndarray, feature_names: list[str]) -> dict`**
- Calls `clf.predict(X_train, pred_contribs=True)` over the full training set
- Returns mean absolute SHAP per feature, sorted descending
- Format: `{"features": [{"name": str, "mean_abs_shap": float, "importance": float}], "computed_at": ISO timestamp, "n_rows": int}`
- Saved to `models/regime_shap_baseline.json` by `train_regime.py` after each train

---

### 2. `scripts/train_regime.py` — baseline snapshot on every train

After saving `models/regime.pkl`, compute and save `models/regime_shap_baseline.json`:
```python
from btc_kalshi_system.models.shap_utils import compute_baseline_snapshot
snapshot = compute_baseline_snapshot(model._clf, X_train, list(_FEATURE_ORDER))
Path("models/regime_shap_baseline.json").write_text(json.dumps(snapshot, indent=2))
```
This gives the monitor an always-current picture of what the model actually learned, updated on every retrain.

---

### 3. `main.py` — live coherence logging at candle close

**Schema migration** — added to `_CANDLE_FEATURES_COLUMN_MIGRATIONS`:
```python
("shap_coherence", "REAL DEFAULT NULL"),
```

**Where coherence is computed:** `RegimeModel.get_regime(features)` already constructs the numpy feature vector internally before calling `clf.predict()`. `shap_coherence` is added to its return dict alongside `prob_up`, `direction`, and `confidence`:
```python
# inside get_regime(), after computing prob_up:
shap_coherence = compute_coherence(self._clf, X)  # X already built
return {"prob_up": ..., "direction": ..., "confidence": ..., "shap_coherence": shap_coherence}
```

`fusion.get_signal()` passes `shap_coherence` through in the `SignalResult`, and `main.py` reads it from the signal at candle close — same pattern as `regime_prob`. No change to the call sites in main.py beyond reading one more field from the signal.

`shap_coherence` is added to the INSERT `col_names` and values list. Placeholder count increments by 1 (36 → 37, accounting for the k15 features already covered by `*vals`).

**Important:** `shap_coherence` is NOT added to `_FEATURE_ORDER`. It is a meta-feature about the model's prediction quality, not a market feature for XGBoost to train on. It lives only in `candle_features` for the calibrator and monitor.

**No backfill.** Historical rows before regime v2 deploy have no model, so `shap_coherence` will be NULL for all pre-deploy rows. This is correct — the score is meaningless without a model. Phase 3c only trains on rows where `regime_prob IS NOT NULL`, so every calibrator training row will have `shap_coherence` populated.

---

### 4. `scripts/regime_v2_monitor.py` — two new outputs in Section 5

**Section 5a: Calibration Curve**

Three fixed confidence tiers based on `|regime_prob - 0.5|`:

| Tier   | Range       | What it means                        |
|--------|-------------|--------------------------------------|
| Low    | < 0.10      | Near-coin-flip, rarely traded        |
| Medium | 0.10 – 0.20 | Typical trade confidence             |
| High   | > 0.20      | Conviction calls, where edge lives   |

For each tier: show `n`, mean predicted confidence, actual win rate (suppressed if n < 10), Brier(regime_prob) vs Brier(kalshi_open_mid).

Key summary line: "Does regime_prob beat Kalshi in the HIGH tier?" — this is the go-live signal. Edge concentrated in high-confidence predictions is the thesis.

**Section 5b: SHAP Feature Contribution (when `regime_shap_baseline.json` exists)**

Shows top-10 features by mean absolute SHAP alongside XGBoost feature importance. This is the honest contextuality picture — what the model actually learned to rely on, updated after every retrain.

Also shows mean `shap_coherence` by confidence tier: "high-confidence candles have coherence X vs low-confidence coherence Y." If the model is working correctly, high-confidence predictions should also have higher coherence.

---

### 5. `scripts/train_calibrator.py` — SHAP coherence as 8th input

Add `shap_coherence` to the training query:
```sql
SELECT regime_prob, signal_edge, deepseek_regime, direction, outcome,
       kronos_raw_15min, brti_volatility_1h, kalshi_spread_normalized,
       shap_coherence FROM (...)
```

Pass to `Calibrator.fit()` alongside existing inputs. NULL rows (pre-SHAP-logging trades) are NULL-guarded in the calibrator's feature pipeline — rows where `shap_coherence IS NULL` should be excluded from the coherence input column rather than filled, preserving the existing 7-input fit for those rows. The `shap_coherence` input is optional per-row, not required.

No changes to `Calibrator` class internals needed if the existing 7-input fit signature accepts an optional 8th argument. If not, extend the signature with `shap_coherence: np.ndarray | None = None`.

---

## What Replaces the Contextuality Check

The contextuality check in `train_regime.py` (the `k15=0.85 in trending vs ranging` probe) is **removed** and replaced with two lines in the dry-run output:

```
── Calibration sanity (training data) ──────────────────────────
  High confidence (|p-0.5|>0.20)  n=89   win_rate=61%  Brier=0.238
  Med  confidence (|p-0.5|<0.10)  n=210  win_rate=51%  Brier=0.271
  Low  confidence (|p-0.5|<0.10)  n=257  win_rate=49%  Brier=0.278
  ✓ Calibration gradient present — confidence correlates with accuracy
```

This is a factual check on real predictions rather than a synthetic probe. It can pass or fail based on data, not assumptions about regime labels.

---

## Files Changed

| File | Change |
|------|--------|
| `btc_kalshi_system/models/shap_utils.py` | New — `compute_coherence` + `compute_baseline_snapshot` |
| `scripts/train_regime.py` | Save `regime_shap_baseline.json` after train; replace contextuality check with calibration sanity |
| `main.py` | Schema migration + `shap_coherence` logged at candle close |
| `scripts/regime_v2_monitor.py` | Section 5: calibration curve + SHAP feature contribution display |
| `scripts/train_calibrator.py` | Add `shap_coherence` to training query and `Calibrator.fit()` call |

---

## What Does NOT Change

- `_FEATURE_ORDER` — `shap_coherence` is not a regime model feature
- `Calibrator` class — no internal changes needed, just an additional input column
- `auto_retrain_calibrator.py` — no changes; calls `train_calibrator.py` as subprocess
- `pretrade_checklist.py` — no gate changes; coherence informs sizing, not blocking

---

## Test Coverage

- `tests/models/test_shap_utils.py` — unit tests for `compute_coherence` (coherence=1.0 when all features agree, =0.5 when perfectly split) and `compute_baseline_snapshot` (output shape, JSON serializable, feature names match)
- `tests/signal/test_feature_order.py` — no change (shap_coherence not in _FEATURE_ORDER)
- `tests/test_auto_retrain_regime.py` — add `shap_coherence` to `_FEATURE_COLS_FOR_DB` fixture
- `tests/models/test_calibrator.py` — extend fit/transform tests to pass `shap_coherence` column

---

## Rollout Order

1. `shap_utils.py` (no dependencies, self-contained)
2. `train_regime.py` update (uses shap_utils, saves baseline snapshot)
3. `main.py` schema + logging (uses shap_utils at candle close)
4. `regime_v2_monitor.py` Section 5 (reads candle_features + baseline snapshot)
5. `train_calibrator.py` update (uses shap_coherence from candle_features)

Phase 3c fires at 500 `regime_prob` rows (~Day 18-20 post regime v2 deploy). Steps 1-4 must complete before that. Step 5 must complete before that.
