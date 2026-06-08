# Regime V2 New Features: k15_kalshi_alignment + k15_delta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two regime features that expose the k15/Kalshi interaction to XGBoost: `k15_kalshi_alignment` (product of k15 and Kalshi deviations from 0.5) and `k15_delta` (one-candle k15 momentum).

**Architecture:** Features are added to `_FEATURE_ORDER` in `regime_model.py`, computed live in `fusion._regime_features()` (for inference), and logged/backfilled in `main.py` `_candle_logger_loop` (for training). The consistency test `test_feature_order_all_three_match` enforces that all three sources stay in sync.

**Tech Stack:** Python 3.11, XGBoost, SQLite, pytest

---

## File Map

| File | Change |
|---|---|
| `btc_kalshi_system/models/regime_model.py` | Add `"k15_kalshi_alignment"` and `"k15_delta"` to `_FEATURE_ORDER` |
| `btc_kalshi_system/signal/fusion.py` | `_k15_delta` init; delta computed before updating `_last_kronos_raw_15min` in `get_signal()`; both features appended in `_regime_features()` |
| `tests/signal/test_feature_order.py` | Update hardcoded count 39 → 41 |
| `main.py` | 2 schema migrations; `_prev_k15_logged` instance var; compute + log both at candle close; one-time backfill SQL on startup |

---

## Task 1: Add features to `_FEATURE_ORDER` (creates failing tests)

**Files:**
- Modify: `btc_kalshi_system/models/regime_model.py`

- [ ] **Step 1: Read `regime_model.py` and find `_FEATURE_ORDER`**

Read `/Users/ezrakornberg/Kronos V2/btc_kalshi_system/models/regime_model.py`. Find the `_FEATURE_ORDER` list — it ends with `"recent_up_fraction"`.

- [ ] **Step 2: Add the two new features at the end of `_FEATURE_ORDER`**

After `"recent_up_fraction",`, add:

```python
    # k15/Kalshi interaction — session 42
    "k15_kalshi_alignment",   # (k15-0.5)*(kalshi_mid-0.5): positive=agrees, negative=disagrees
    "k15_delta",              # k15_now - k15_prior_candle: near-zero=stalling, nonzero=fresh signal
```

- [ ] **Step 3: Run the feature order tests to confirm they now fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/signal/test_feature_order.py -v 2>&1 | tail -20
```

Expected: `test_feature_order_all_three_match` FAILS (count 39 != 41) and `test_feature_order_regime_model_vs_fusion` FAILS (fusion missing new keys). Two failures.

---

## Task 2: Add features to fusion.py

**Files:**
- Modify: `btc_kalshi_system/signal/fusion.py`

- [ ] **Step 1: Read fusion.py init and get_signal()**

Read `/Users/ezrakornberg/Kronos V2/btc_kalshi_system/signal/fusion.py` lines 110–180.

- [ ] **Step 2: Add `_k15_delta` to `__init__`**

In `SignalFusionEngine.__init__`, after `self._last_deepseek_dir_prob: float = 0.5`, add:

```python
        self._k15_delta: float | None = None
```

- [ ] **Step 3: Compute delta in `get_signal()` before updating `_last_kronos_raw_15min`**

Find line `self._last_kronos_raw_15min = kronos_raw_15min` (around line 168). Replace it with:

```python
        # k15_delta: change from prior k15 (kronos stall detection)
        if kronos_raw_15min is not None and self._last_kronos_raw_15min is not None:
            self._k15_delta = round(kronos_raw_15min - self._last_kronos_raw_15min, 4)
        else:
            self._k15_delta = None
        self._last_kronos_raw_15min = kronos_raw_15min
```

- [ ] **Step 4: Add both features to `_regime_features()` return dict**

Read the `_regime_features()` method. Find the end of the returned `features` dict (the last entry is `"recent_up_fraction": recent_up_fraction`). Add immediately after it (still inside the dict, before the closing `}`):

```python
            # k15/Kalshi interaction features — session 42
            "k15_kalshi_alignment": (
                round(
                    (self._last_kronos_raw_15min - 0.5)
                    * (self._market_context.get("kalshi_mid_cents", None) / 100.0 - 0.5),
                    4,
                )
                if self._last_kronos_raw_15min is not None
                and self._market_context.get("kalshi_mid_cents") is not None
                else None
            ),
            "k15_delta": self._k15_delta,
```

- [ ] **Step 5: Run feature order tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/signal/test_feature_order.py -v 2>&1 | tail -20
```

Expected: `test_feature_order_regime_model_vs_fusion` now PASSES. `test_feature_order_all_three_match` still FAILS (count 39 != 41).

- [ ] **Step 6: Commit fusion changes**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/models/regime_model.py btc_kalshi_system/signal/fusion.py && git commit -m "$(cat <<'EOF'
feat: add k15_kalshi_alignment + k15_delta to regime _FEATURE_ORDER and fusion

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Fix feature order test count

**Files:**
- Modify: `tests/signal/test_feature_order.py`

- [ ] **Step 1: Update the hardcoded count**

Find `assert len(_FEATURE_ORDER) == 39` in `test_feature_order_all_three_match`. Change to:

```python
    assert len(_FEATURE_ORDER) == 41  # added k15_kalshi_alignment, k15_delta (session 42)
```

- [ ] **Step 2: Run all feature order tests**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/signal/test_feature_order.py -v 2>&1 | tail -15
```

Expected: all 4 feature order tests PASS.

- [ ] **Step 3: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add tests/signal/test_feature_order.py && git commit -m "$(cat <<'EOF'
test: update feature order count 39→41 for k15_kalshi_alignment + k15_delta

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: main.py — schema + logging + backfill

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add schema migrations**

In `_CANDLE_FEATURES_COLUMN_MIGRATIONS` (around line 325), find `("mid_candle_model_prob", "REAL DEFAULT NULL")`. Add immediately after it:

```python
    ("k15_kalshi_alignment",  "REAL DEFAULT NULL"),
    ("k15_delta",             "REAL DEFAULT NULL"),
```

- [ ] **Step 2: Add `_prev_k15_logged` instance var to `__init__`**

Find `self._candle_open_brti: dict[str, float | None] = {}` (around line 452). Add immediately after it:

```python
        self._prev_k15_logged: float | None = None
```

- [ ] **Step 3: Add backfill SQL after `self._db.commit()` in `__init__`**

Find the `self._db.commit()` that follows all the `ALTER TABLE` migration loops (around line 438). Add immediately after it:

```python
        # One-time backfill for k15_kalshi_alignment and k15_delta on historical rows.
        self._db.execute("""
            UPDATE candle_features
            SET k15_kalshi_alignment = ROUND((kronos_raw_15min - 0.5) * (kalshi_open_mid - 0.5), 4)
            WHERE kronos_raw_15min IS NOT NULL
              AND kalshi_open_mid IS NOT NULL
              AND k15_kalshi_alignment IS NULL
        """)
        self._db.execute("""
            UPDATE candle_features
            SET k15_delta = ROUND(kronos_raw_15min - (
                SELECT prev.kronos_raw_15min FROM candle_features prev
                WHERE prev.candle_ts < candle_features.candle_ts
                  AND prev.kronos_raw_15min IS NOT NULL
                ORDER BY prev.candle_ts DESC LIMIT 1
            ), 4)
            WHERE kronos_raw_15min IS NOT NULL
              AND k15_delta IS NULL
        """)
        self._db.commit()
```

- [ ] **Step 4: Compute both features at candle close in `_candle_logger_loop`**

In `_candle_logger_loop`, find the block that computes `kalshi_early_drift` (around line 879). After the `kalshi_early_drift` computation block (after its closing `)`), add:

```python
                k15_kalshi_alignment = None
                k15_delta = None
                _k15_now = features.get("kronos_raw_15min")
                if _k15_now is not None and kalshi_open_mid is not None:
                    k15_kalshi_alignment = round(
                        (_k15_now - 0.5) * (kalshi_open_mid - 0.5), 4
                    )
                if _k15_now is not None and self._prev_k15_logged is not None:
                    k15_delta = round(_k15_now - self._prev_k15_logged, 4)
                if _k15_now is not None:
                    self._prev_k15_logged = _k15_now
```

- [ ] **Step 5: Add columns to the INSERT col_names**

Find the `col_names` string in the INSERT block (around line 887). Change:

```python
                    "spread_change, oi_delta_at_midcandle, k5_candle_ts, mid_candle_model_prob, "
```

To:

```python
                    "spread_change, oi_delta_at_midcandle, k5_candle_ts, mid_candle_model_prob, "
                    "k15_kalshi_alignment, k15_delta, "
```

- [ ] **Step 6: Add values to the INSERT VALUES list and update placeholder count**

Find `mid_candle_model_prob,` in the VALUES list (around line 941). Add immediately after it:

```python
                        k15_kalshi_alignment,
                        k15_delta,
```

Update the placeholder count from `36` to `38`:

```python
                placeholders = ", ".join(["?"] * (38 + len(cols)))
```

- [ ] **Step 7: Verify main.py parses**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -c "import main; print('OK')"
```

Expected: `OK`.

---

## Task 5: Full test suite + restart + push

- [ ] **Step 1: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -5
```

Expected: 599+ passing, 0 failures.

- [ ] **Step 2: Commit main.py**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add main.py && git commit -m "$(cat <<'EOF'
feat: log k15_kalshi_alignment + k15_delta to candle_features + backfill

Schema migrations, candle-close logging, one-time backfill SQL for
historical rows. Regime v2 will pick up both features on next retrain.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Restart service**

```bash
launchctl kickstart -k gui/$(id -u)/com.kronos.v2
```

- [ ] **Step 4: Verify backfill ran**

```bash
sleep 15 && cd "/Users/ezrakornberg/Kronos V2" && python3 -c "
import sqlite3
conn = sqlite3.connect('trades.db')
total = conn.execute('SELECT COUNT(*) FROM candle_features WHERE features_stale=0').fetchone()[0]
filled_align = conn.execute('SELECT COUNT(*) FROM candle_features WHERE k15_kalshi_alignment IS NOT NULL').fetchone()[0]
filled_delta = conn.execute('SELECT COUNT(*) FROM candle_features WHERE k15_delta IS NOT NULL').fetchone()[0]
sample = conn.execute('SELECT kronos_raw_15min, kalshi_open_mid, k15_kalshi_alignment, k15_delta FROM candle_features WHERE k15_kalshi_alignment IS NOT NULL ORDER BY candle_ts DESC LIMIT 3').fetchall()
print(f'Total rows: {total}  k15_kalshi_alignment filled: {filled_align}  k15_delta filled: {filled_delta}')
for r in sample:
    print(f'  k15={r[0]:.3f} kalshi={r[1]:.3f} alignment={r[2]:.4f} delta={r[3]}')
conn.close()
"
```

Expected: 350+ rows filled for `k15_kalshi_alignment`, 340+ for `k15_delta`.

- [ ] **Step 5: Push**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git push origin main
```
