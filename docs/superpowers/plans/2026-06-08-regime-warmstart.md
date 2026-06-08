# Regime Warm-Start + Faster Row Trigger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop the row retrain trigger from +200 to +50 rows (~12h cadence) and add XGBoost warm-start so retrains continue from existing trees rather than starting from scratch.

**Architecture:** `RegimeModel.train()` gains an optional `warm_start_from` param that passes the existing booster to `fit()`, adding 25 trees instead of training 100 from scratch. `auto_retrain_regime.py` uses warm-start for ROW and REGIME-SHIFT triggers (incremental) and cold-start for TIME-BASED and FORCE (full resets). The holdout Brier guard is unchanged. `train_regime.py` gets a `--warm-start` flag for manual use.

**Tech Stack:** Python 3.11, XGBoost (sklearn API + native booster), joblib, pytest

---

## File Map

| File | Change |
|---|---|
| `btc_kalshi_system/models/regime_model.py` | `train()` gains `warm_start_from: RegimeModel \| None = None` |
| `scripts/auto_retrain_regime.py` | `_ROW_TRIGGER_DELTA` 200→50; new `_use_warm_start()` helper; warm-start wired in `main()` |
| `scripts/train_regime.py` | `--warm-start` argparse flag |
| `tests/models/test_regime_model.py` | 3 new warm-start tests |
| `tests/test_auto_retrain_regime.py` | 4 new tests: row delta constant, `_use_warm_start()` logic |

---

## Task 1: Warm-start tests for RegimeModel (write failing tests)

**Files:**
- Modify: `tests/models/test_regime_model.py`

- [ ] **Step 1: Append 3 tests at the end of the file**

```python
# ── warm-start ─────────────────────────────────────────────────────────────────

def test_cold_start_produces_100_trees():
    """Default cold start always uses 100 boosted rounds."""
    X, y = _synthetic_features()
    model = RegimeModel()
    model.train(X, y)
    assert model._clf.get_booster().num_boosted_rounds() == 100


def test_warm_start_adds_25_trees():
    """Warm-start from existing model produces base + 25 total boosted rounds."""
    X, y = _synthetic_features()
    base = RegimeModel()
    base.train(X, y)
    base_rounds = base._clf.get_booster().num_boosted_rounds()  # 100

    warm = RegimeModel()
    warm.train(X, y, warm_start_from=base)
    assert warm._clf.get_booster().num_boosted_rounds() == base_rounds + 25


def test_warm_start_none_is_cold_start():
    """warm_start_from=None is identical to cold start (100 trees)."""
    X, y = _synthetic_features()
    model = RegimeModel()
    model.train(X, y, warm_start_from=None)
    assert model._clf.get_booster().num_boosted_rounds() == 100
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/models/test_regime_model.py::test_cold_start_produces_100_trees tests/models/test_regime_model.py::test_warm_start_adds_25_trees tests/models/test_regime_model.py::test_warm_start_none_is_cold_start -v 2>&1 | tail -15
```

Expected: `test_warm_start_adds_25_trees` FAILS (`train()` has no `warm_start_from` param). The other two may pass or fail — confirm `test_warm_start_adds_25_trees` fails.

---

## Task 2: Implement warm_start_from in RegimeModel.train()

**Files:**
- Modify: `btc_kalshi_system/models/regime_model.py`

- [ ] **Step 1: Read the file**

Read `/Users/ezrakornberg/Kronos V2/btc_kalshi_system/models/regime_model.py` to confirm the current `train()` signature.

- [ ] **Step 2: Replace train() with warm-start version**

Find the `train()` method (line ~98). Replace it entirely with:

```python
    def train(self, X: np.ndarray, y: np.ndarray,
              warm_start_from: "RegimeModel | None" = None,
              **xgb_kwargs) -> "RegimeModel":
        """
        Fit the XGBoost classifier.

        warm_start_from: if provided, continues boosting from that model's
        trees (+25 rounds). Use for incremental retrains (ROW-BASED,
        REGIME-SHIFT). Pass None for a full cold start (100 rounds).

        Extra keyword arguments (e.g. scale_pos_weight) are forwarded to
        XGBClassifier and override the defaults below.
        """
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

- [ ] **Step 3: Run the 3 warm-start tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/models/test_regime_model.py::test_cold_start_produces_100_trees tests/models/test_regime_model.py::test_warm_start_adds_25_trees tests/models/test_regime_model.py::test_warm_start_none_is_cold_start -v 2>&1 | tail -15
```

Expected: all 3 PASS.

- [ ] **Step 4: Run full regime model test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/models/test_regime_model.py -v 2>&1 | tail -20
```

Expected: all tests pass (no regressions).

- [ ] **Step 5: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/models/regime_model.py tests/models/test_regime_model.py && git commit -m "$(cat <<'EOF'
feat: RegimeModel.train() warm_start_from parameter (+25 trees)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Tests for row trigger delta + _use_warm_start() (write failing tests)

**Files:**
- Modify: `tests/test_auto_retrain_regime.py`

- [ ] **Step 1: Add _use_warm_start to imports and append 5 tests**

In `tests/test_auto_retrain_regime.py`, find the import block (lines 14–31). Add `_use_warm_start` to the `from scripts.auto_retrain_regime import (...)` block:

```python
from scripts.auto_retrain_regime import (
    ...
    _REGIME_PAUSE_FLAG,
    _use_warm_start,       # ADD THIS LINE
)
```

Then append these 5 tests at the end of the file:

```python
# ── row trigger delta ──────────────────────────────────────────────────────────

def test_row_trigger_delta_is_50():
    """Row trigger fires every 50 new rows (~12h at 96 candles/day)."""
    assert _ROW_TRIGGER_DELTA == 50


# ── _use_warm_start ────────────────────────────────────────────────────────────

def test_use_warm_start_true_for_row_based():
    assert _use_warm_start("ROW-BASED") is True


def test_use_warm_start_true_for_regime_shift():
    assert _use_warm_start("REGIME-SHIFT (0.45→0.60)") is True


def test_use_warm_start_false_for_time_based():
    assert _use_warm_start("TIME-BASED") is False


def test_use_warm_start_false_for_force():
    assert _use_warm_start("FORCE") is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/test_auto_retrain_regime.py::test_row_trigger_delta_is_50 tests/test_auto_retrain_regime.py::test_use_warm_start_true_for_row_based -v 2>&1 | tail -15
```

Expected: both FAIL — `_use_warm_start` doesn't exist yet and `_ROW_TRIGGER_DELTA` is still 200.

---

## Task 4: Implement row trigger change + warm-start wiring

**Files:**
- Modify: `scripts/auto_retrain_regime.py`

- [ ] **Step 1: Read the file first**

Read `/Users/ezrakornberg/Kronos V2/scripts/auto_retrain_regime.py`, focusing on lines 1-55 (constants + imports) and lines 310-370 (training call in `main()`).

- [ ] **Step 2: Change `_ROW_TRIGGER_DELTA` and update comment**

Find line ~44:
```python
_ROW_TRIGGER_DELTA = 200   # retrain when +200 new candle rows since last train (~2 days)
```

Replace with:
```python
_ROW_TRIGGER_DELTA = 50    # retrain when +50 new candle rows since last train (~12h)
```

Also update the header comment at line ~6 from:
```
#   2. Row-based: +200 new candle_features rows since last train (~2 days at 96/day)
```
to:
```
#   2. Row-based: +50 new candle_features rows since last train (~12h at 96/day)
```

- [ ] **Step 3: Add `_use_warm_start()` helper**

Add this function immediately after the `should_deploy()` function (around line 230):

```python
def _use_warm_start(trigger: str) -> bool:
    """True for incremental triggers (ROW-BASED, REGIME-SHIFT); False for full resets."""
    return trigger.startswith("ROW-BASED") or trigger.startswith("REGIME-SHIFT")
```

- [ ] **Step 4: Wire warm-start into main() training call**

Find the training block in `main()` (around line 310–325):
```python
    candidate = RegimeModel()
    candidate.train(X_train, y_train, **extra_kwargs)
```

Replace with:
```python
    # Warm-start for incremental triggers; cold-start for TIME-BASED and FORCE.
    deployed_for_warm = None
    if _use_warm_start(trigger):
        try:
            deployed_for_warm = RegimeModel.load(str(args.out))
        except FileNotFoundError:
            pass  # no model yet — cold start

    candidate = RegimeModel()
    candidate.train(X_train, y_train, warm_start_from=deployed_for_warm, **extra_kwargs)
```

- [ ] **Step 5: Run the 5 new auto_retrain tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/test_auto_retrain_regime.py::test_row_trigger_delta_is_50 tests/test_auto_retrain_regime.py::test_use_warm_start_true_for_row_based tests/test_auto_retrain_regime.py::test_use_warm_start_true_for_regime_shift tests/test_auto_retrain_regime.py::test_use_warm_start_false_for_time_based tests/test_auto_retrain_regime.py::test_use_warm_start_false_for_force -v 2>&1 | tail -15
```

Expected: all 5 PASS.

- [ ] **Step 6: Run full auto_retrain test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/test_auto_retrain_regime.py -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add scripts/auto_retrain_regime.py tests/test_auto_retrain_regime.py && git commit -m "$(cat <<'EOF'
feat: row trigger 200→50 + warm-start wiring in auto_retrain_regime

ROW-BASED and REGIME-SHIFT triggers continue from existing trees (+25);
TIME-BASED and FORCE always cold-start (100 trees, resets tree count).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: --warm-start flag in train_regime.py

**Files:**
- Modify: `scripts/train_regime.py`

- [ ] **Step 1: Read parse_args() in train_regime.py**

Search for `parse_args` in `/Users/ezrakornberg/Kronos V2/scripts/train_regime.py` to find the argument block and where the training call is.

- [ ] **Step 2: Add --warm-start to parse_args()**

Find the `parse_args()` function. After the existing `--force` argument, add:

```python
    p.add_argument("--warm-start", action="store_true",
                   help="Continue training from existing regime.pkl (+25 trees). "
                        "Falls back to cold start if no model found.")
```

- [ ] **Step 3: Wire warm-start into the training call**

Find the section in `main()` where `model = RegimeModel()` and `model.train(X_train, y_train, **extra_kwargs)` are called (around line 201). Just before those two lines, add:

```python
    warm_start_model = None
    if args.warm_start:
        try:
            warm_start_model = RegimeModel.load(args.out)
            n_existing = warm_start_model._clf.get_booster().num_boosted_rounds()
            print(f"Warm-start: loaded {args.out} ({n_existing} trees → +25)")
        except FileNotFoundError:
            print(f"Warm-start: no model at {args.out} — cold start (100 trees)")
```

Then change the `.train()` call to pass `warm_start_from=warm_start_model`:

```python
    model = RegimeModel()
    model.train(X_train, y_train, warm_start_from=warm_start_model, **extra_kwargs)
```

- [ ] **Step 4: Verify parse**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 scripts/train_regime.py --help 2>&1 | grep "warm"
```

Expected: `--warm-start` appears in the help output.

- [ ] **Step 5: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add scripts/train_regime.py && git commit -m "$(cat <<'EOF'
feat: --warm-start flag for train_regime.py

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Full test suite + push

- [ ] **Step 1: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: 593+ passing, 0 failures (585 existing + 3 regime model + 5 auto_retrain).

- [ ] **Step 2: Push**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git push origin main
```
