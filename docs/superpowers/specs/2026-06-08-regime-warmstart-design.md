# Regime Model Warm-Start + Faster Row Trigger Design — 2026-06-08

## Goal

Two improvements to regime model retraining: (1) halve the row trigger from +200 to +50 so the model learns from new data every ~12h instead of every 2 days; (2) add XGBoost warm-start so retrains continue from existing trees rather than starting from scratch.

---

## Change 1: Row Trigger 200 → 50

**File:** `scripts/auto_retrain_regime.py`

```python
_ROW_TRIGGER_DELTA = 50   # was 200 — retrain ~every 12h instead of every 2 days
```

Update the comment above to reflect the new cadence (~12h at 96 candles/day).

---

## Change 2: XGBoost Warm-Start

### Mechanism

XGBoost's `fit(xgb_model=existing_booster)` continues boosting from existing trees. New trees are fit on the full dataset (not just new rows) starting from the existing model's residuals. Old knowledge is preserved in existing trees; new patterns are captured in the additional trees.

### `RegimeModel.train()` changes

Add `warm_start_from: RegimeModel | None = None` parameter:

```python
def train(self, X: np.ndarray, y: np.ndarray,
          warm_start_from: "RegimeModel | None" = None,
          **xgb_kwargs) -> "RegimeModel":
    defaults = {
        "n_estimators": 25 if warm_start_from is not None else 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
        "random_state": 42,
    }
    defaults.update(xgb_kwargs)
    self._clf = xgb.XGBClassifier(**defaults)
    if warm_start_from is not None:
        self._clf.fit(X, y, xgb_model=warm_start_from._clf.get_booster())
    else:
        self._clf.fit(X, y)
    return self
```

- Cold start (default): 100 trees — behavior unchanged
- Warm start: 25 additional trees on top of existing booster

### `auto_retrain_regime.py` changes

- **ROW trigger** (fires every ~12h): warm-start from deployed model
- **REGIME_SHIFT trigger**: warm-start from deployed model (rapid adaptation to market shift)
- **TIME trigger** (14 days): cold-start — mandatory tree count reset; prevents unbounded growth

Pass `warm_start_from` to the candidate training call. If `regime.pkl` doesn't exist (first deploy), always cold-start regardless of trigger type.

### `train_regime.py` changes

Add `--warm-start` flag:

```bash
python3 scripts/train_regime.py --warm-start  # loads existing regime.pkl, adds 25 trees
python3 scripts/train_regime.py               # cold start, 100 trees (default)
```

If `--warm-start` is set but `models/regime.pkl` doesn't exist, fall back to cold start with a warning.

---

## Holdout Guard

Unchanged. Warm-start candidate must achieve strictly lower Brier than the deployed model on the same 100-row holdout before it saves. If warm-start doesn't improve (e.g., new rows are noisy), the deployed model stays.

---

## Tree Count Growth

Over a 14-day cycle:
- ~27 ROW triggers × 25 trees = 675 additional trees
- TIME trigger resets to 100 trees

Peak tree count before reset: ~775. XGBoost prediction time is O(trees) but at this scale is negligible (<1ms). No explicit cap needed.

---

## Tests

| Test | What it checks |
|---|---|
| `test_regime_model_warm_start_adds_trees` | `warm_start_from` produces a model with more estimators than cold-start |
| `test_regime_model_cold_start_unchanged` | Cold start still produces 100-tree model (no regression) |
| `test_regime_model_warm_start_none_fallback` | `warm_start_from=None` is equivalent to cold start |
| `test_auto_retrain_row_trigger_uses_warm_start` | ROW trigger passes `warm_start_from` when model exists |
| `test_auto_retrain_time_trigger_cold_starts` | TIME trigger does NOT pass `warm_start_from` |

---

## Files Changed

| File | Change |
|---|---|
| `btc_kalshi_system/models/regime_model.py` | `train()` gains `warm_start_from` param |
| `scripts/auto_retrain_regime.py` | `_ROW_TRIGGER_DELTA` 200→50; warm-start wiring for ROW/SHIFT triggers |
| `scripts/train_regime.py` | `--warm-start` flag |
| `tests/models/test_regime_model.py` | 3 new tests |
| `tests/test_auto_retrain_regime.py` | 2 new tests |
