# Gate 15 + Mid-Candle Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `MidCandleModel` into the live trading loop — load at startup, score at mid-candle snapshot time, write to Redis, restructure Gate 12 for the 40-60% entry window, and add Gate 15 with cross-candle contamination guard.

**Architecture:** Two files change. `pretrade_checklist.py` gains a `_mid_candle_model_loaded` flag, a restructured Gate 12 (mid-window bypass), and a new Gate 15 (reads Redis, validates `candle_ts`). `main.py` loads the model at startup, scores in `_candle_logger_loop`, writes `mid_candle:prob` to Redis, persists the score to the DB, and passes `current_candle_ts` to both `checklist.run()` calls. All changes are inert until `models/mid_candle.pkl` exists.

**Tech Stack:** Python 3.11, XGBoost (`MidCandleModel`), Redis, SQLite, pytest

---

## File Map

| File | What changes |
|---|---|
| `btc_kalshi_system/execution/pretrade_checklist.py` | `import json`, `_mid_candle_model_loaded` attr, `current_candle_ts` param, Gate 12 restructure, Gate 15 |
| `main.py` | Import `MidCandleModel`, model load in `__init__`, schema column, `_candle_logger_loop` scoring + Redis write, `mid_candle_model_prob` in INSERT, `_open_dt` init + `_current_candle_ts` derivation, both `checklist.run()` calls updated |
| `tests/execution/test_pretrade_checklist.py` | 5 new tests |

---

## Task 1: Gate 12 mid-window tests (write failing tests)

**Files:**
- Modify: `tests/execution/test_pretrade_checklist.py`

- [ ] **Step 1: Add imports and two Gate 12 mid-window tests**

Open `tests/execution/test_pretrade_checklist.py`. Add `import json` at the top (after existing imports). Append these two tests at the end of the file:

```python
def test_gate12_mid_window_no_model_blocks(checklist):
    """Progress 50% but model not loaded → Gate 12 blocks with 'not loaded' message."""
    signal = make_signal(regime_features={"candle_progress": 0.50})
    r = checklist.run(**base_kwargs(signal))
    assert not r.passed
    assert r.failed_gate == 12
    assert "not loaded" in r.failed_reason


def test_gate12_mid_window_with_model_passes(checklist):
    """Progress 50% and model loaded → Gate 12 passes (falls through to Gate 15)."""
    checklist._mid_candle_model_loaded = True
    signal = make_signal(regime_features={"candle_progress": 0.50})
    r = checklist.run(**base_kwargs(signal))
    assert r.failed_gate != 12
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/execution/test_pretrade_checklist.py::test_gate12_mid_window_no_model_blocks tests/execution/test_pretrade_checklist.py::test_gate12_mid_window_with_model_passes -v 2>&1 | tail -20
```

Expected: both FAIL — `test_gate12_mid_window_no_model_blocks` fails because progress=0.50 currently passes Gate 12 (it's under the 10% cap... wait, 0.50 > 0.10 cap so it DOES fail Gate 12 currently, but for the wrong reason — "exceeds cap", not "not loaded"). `test_gate12_mid_window_with_model_passes` fails because there's no `_mid_candle_model_loaded` attribute and progress=0.50 exceeds the cap.

---

## Task 2: Gate 12 restructure (make tests pass)

**Files:**
- Modify: `btc_kalshi_system/execution/pretrade_checklist.py`

- [ ] **Step 1: Add `json` import**

At the top of `pretrade_checklist.py`, add `import json` after `from dataclasses import dataclass`:

```python
from dataclasses import dataclass
import json
from typing import Optional
```

- [ ] **Step 2: Add `_mid_candle_model_loaded` to `__init__` and `current_candle_ts` to `run()`**

In `PreTradeChecklist.__init__`, add after `self._redis = redis.from_url(config.REDIS_URL)`:

```python
        self._mid_candle_model_loaded: bool = False
```

In `PreTradeChecklist.run()`, add `current_candle_ts: Optional[str] = None` as the last parameter:

```python
    def run(
        self,
        signal: TradingSignal,
        best_ask_cents: int,
        best_bid_cents: int,
        available_contracts: int,
        current_exposure: float,
        same_timeframe_open: bool,
        composite_price: float,
        edge_above_threshold: bool,
        fresh_kalshi_mid: float = 0.5,
        is_drifting: bool = False,
        direction_win_rate: Optional[float] = None,
        is_bootstrap: bool = False,
        current_candle_ts: Optional[str] = None,
    ) -> ChecklistResult:
```

- [ ] **Step 3: Restructure Gate 12**

Find the current Gate 12 block (starts with `# Gate 12 — Dynamic candle progress window`). Replace the two `if candle_progress < _PROGRESS_FLOOR` / `if candle_progress > _cap` checks with the mid-window restructure. The full Gate 12 block should become:

```python
        # Gate 12 — Dynamic candle progress window (floor 3%, ceiling 5-10%; mid-window bypass at 40-60%)
        # Floor: wait for T+27s so Kalshi can reprice to the candle open (avg 2.71¢ move in 30s).
        # Ceiling: edge decays rapidly after 90s (10-15% = -$2.62/trade, 15-20% = -$5.74/trade).
        # Thresholds: _HIGH_VOL=0.3%/5min, _WIDE_SPREAD=4¢. Rules: both→5%, else→10%.
        # Mid-window (40-60%): allowed only when mid_candle_model is loaded; Gate 15 confirms direction.
        _PROGRESS_FLOOR = 0.03
        candle_progress = (signal.regime_features or {}).get("candle_progress", 0.0) or 0.0
        _volatility  = (signal.regime_features or {}).get("brti_volatility_1h", 0.0) or 0.0
        _spread      = (signal.market_context  or {}).get("kalshi_spread_normalized", 0.0) or 0.0
        _vol_ratio   = (signal.regime_features or {}).get("volume_ratio_1h", 1.0) or 1.0
        _cap = _PROGRESS_CAP_MODEL.get_cap(_volatility, _spread, _vol_ratio)
        _IN_MID_WINDOW = 0.40 <= candle_progress <= 0.60
        if not _IN_MID_WINDOW:
            if candle_progress < _PROGRESS_FLOOR:
                return fail(12, f"Candle progress {candle_progress:.3f} below {_PROGRESS_FLOOR} floor — waiting for T+27s Kalshi reaction")
            if candle_progress > _cap:
                return fail(12, (
                    f"Candle progress {candle_progress:.2f} exceeds dynamic cap {_cap:.2f} "
                    f"(vol={_volatility:.4f} spread={_spread:.3f})"
                ))
        elif not self._mid_candle_model_loaded:
            return fail(12, "Mid-candle window (40-60%) requires model — not loaded yet")
        # else: mid-window + model loaded → fall through to Gate 15
```

- [ ] **Step 4: Run Gate 12 tests to verify they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/execution/test_pretrade_checklist.py::test_gate12_mid_window_no_model_blocks tests/execution/test_pretrade_checklist.py::test_gate12_mid_window_with_model_passes -v 2>&1 | tail -20
```

Expected: both PASS.

- [ ] **Step 5: Run full checklist test suite to verify no regressions**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/execution/test_pretrade_checklist.py tests/execution/test_progress_cap_model.py -v 2>&1 | tail -30
```

Expected: all existing tests pass.

- [ ] **Step 6: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/execution/pretrade_checklist.py tests/execution/test_pretrade_checklist.py && git commit -m "$(cat <<'EOF'
feat: Gate 12 mid-window bypass + _mid_candle_model_loaded flag

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Gate 15 tests (write failing tests)

**Files:**
- Modify: `tests/execution/test_pretrade_checklist.py`

- [ ] **Step 1: Append five Gate 15 tests**

Add at the end of `tests/execution/test_pretrade_checklist.py`:

```python
def test_gate15_stale_candle_ts_skips(checklist):
    """Gate 15 skips silently when Redis candle_ts doesn't match current candle."""
    checklist._mid_candle_model_loaded = True

    def _get_side_effect(key):
        if key == "mid_candle:prob":
            return json.dumps(
                {"prob": 0.20, "candle_ts": "2026-01-01T00:00:00+00:00"}
            ).encode()
        return None  # trading:loss_streak → 0

    checklist._redis.get.side_effect = _get_side_effect
    signal = make_signal(regime_features={"candle_progress": 0.50})
    r = checklist.run(
        **base_kwargs(signal),
        current_candle_ts="2026-01-01T00:15:00+00:00",  # different from Redis ts
    )
    assert r.passed  # stale ts → gate skipped → trade allowed


def test_gate15_bearish_blocks_yes(checklist):
    """Gate 15 blocks YES when mid-candle model prob < 0.38."""
    checklist._mid_candle_model_loaded = True
    current_ts = "2026-01-01T00:15:00+00:00"
    checklist._redis.get.return_value = json.dumps(
        {"prob": 0.25, "candle_ts": current_ts}
    ).encode()
    signal = make_signal(direction=1, regime_features={"candle_progress": 0.50})
    r = checklist.run(**base_kwargs(signal), current_candle_ts=current_ts)
    assert not r.passed
    assert r.failed_gate == 15
    assert "bearish" in r.failed_reason


def test_gate15_bullish_blocks_no(checklist):
    """Gate 15 blocks NO when mid-candle model prob > 0.62."""
    checklist._mid_candle_model_loaded = True
    current_ts = "2026-01-01T00:15:00+00:00"
    checklist._redis.get.return_value = json.dumps(
        {"prob": 0.75, "candle_ts": current_ts}
    ).encode()
    signal = make_signal(direction=0, regime_features={"candle_progress": 0.50})
    r = checklist.run(**base_kwargs(signal), current_candle_ts=current_ts)
    assert not r.passed
    assert r.failed_gate == 15
    assert "bullish" in r.failed_reason


def test_gate15_neutral_prob_allows_trade(checklist):
    """Gate 15 allows trade when prob is within neutral band (0.38-0.62)."""
    checklist._mid_candle_model_loaded = True
    current_ts = "2026-01-01T00:15:00+00:00"
    checklist._redis.get.return_value = json.dumps(
        {"prob": 0.55, "candle_ts": current_ts}  # neutral — not bearish enough to block YES
    ).encode()
    signal = make_signal(direction=1, regime_features={"candle_progress": 0.50})
    r = checklist.run(**base_kwargs(signal), current_candle_ts=current_ts)
    assert r.failed_gate != 15


def test_gate15_no_redis_score_allows_trade(checklist):
    """Gate 15 skips when Redis key is absent (model loaded but no score yet)."""
    checklist._mid_candle_model_loaded = True
    checklist._redis.get.return_value = None  # no score in Redis
    signal = make_signal(regime_features={"candle_progress": 0.50})
    r = checklist.run(**base_kwargs(signal))
    assert r.failed_gate != 15
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/execution/test_pretrade_checklist.py::test_gate15_stale_candle_ts_skips tests/execution/test_pretrade_checklist.py::test_gate15_bearish_blocks_yes tests/execution/test_pretrade_checklist.py::test_gate15_bullish_blocks_no -v 2>&1 | tail -20
```

Expected: all FAIL — Gate 15 doesn't exist yet.

---

## Task 4: Gate 15 implementation (make tests pass)

**Files:**
- Modify: `btc_kalshi_system/execution/pretrade_checklist.py`

- [ ] **Step 1: Add Gate 15 immediately after the Gate 12 block**

Find the end of the Gate 12 block (the `# else: mid-window + model loaded → fall through to Gate 15` comment). Insert Gate 15 immediately after that comment:

```python
        # Gate 15 — Mid-candle model direction filter (40-60% window only)
        # Only fires when candle is in mid-window. Skips silently if Redis score is
        # from a different candle (cross-candle contamination guard).
        if 0.40 <= candle_progress <= 0.60:
            _mc_raw = self._redis.get("mid_candle:prob")
            if _mc_raw:
                _mc = json.loads(_mc_raw)
                if _mc.get("candle_ts") == current_candle_ts:
                    _mc_prob = _mc.get("prob")
                    if _mc_prob is not None:
                        if signal.direction == 1 and _mc_prob < 0.38:
                            return fail(15, f"Mid-candle model bearish ({_mc_prob:.2f}) vs YES entry")
                        if signal.direction == 0 and _mc_prob > 0.62:
                            return fail(15, f"Mid-candle model bullish ({_mc_prob:.2f}) vs NO entry")
```

- [ ] **Step 2: Run Gate 15 tests to verify they pass**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/execution/test_pretrade_checklist.py -k "gate15" -v 2>&1 | tail -20
```

Expected: all 5 Gate 15 tests PASS.

- [ ] **Step 3: Run full checklist test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest tests/execution/test_pretrade_checklist.py tests/execution/test_progress_cap_model.py -v 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add btc_kalshi_system/execution/pretrade_checklist.py tests/execution/test_pretrade_checklist.py && git commit -m "$(cat <<'EOF'
feat: Gate 15 mid-candle model direction filter with candle_ts validation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: main.py — model load at startup + schema column

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add MidCandleModel import**

Find the imports block in `main.py` where other model imports live (search for `from btc_kalshi_system.models`). Add:

```python
from btc_kalshi_system.models.mid_candle_model import MidCandleModel
```

- [ ] **Step 2: Add schema column to `_CANDLE_FEATURES_COLUMN_MIGRATIONS`**

In `_CANDLE_FEATURES_COLUMN_MIGRATIONS` (around line 258), find the entry for `k5_candle_ts` and add `mid_candle_model_prob` immediately after it:

```python
    ("k5_candle_ts",              "TEXT DEFAULT NULL"),
    ("mid_candle_model_prob",     "REAL DEFAULT NULL"),
```

- [ ] **Step 3: Load model in `__init__`**

In `KronosV2.__init__`, find `self._checklist = PreTradeChecklist(self._kelly)` (around line 359). Immediately after it, add:

```python
        try:
            self._mid_candle_model = MidCandleModel.load(config.MID_CANDLE_MODEL_PATH)
            self._checklist._mid_candle_model_loaded = True
            logger.info(f"MidCandleModel loaded from {config.MID_CANDLE_MODEL_PATH}")
        except FileNotFoundError:
            self._mid_candle_model = None
```

- [ ] **Step 4: Verify import and startup changes parse correctly**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -c "import main; print('OK')"
```

Expected: `OK` (no import errors).

---

## Task 6: main.py — `_candle_logger_loop` scoring + Redis write + INSERT

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Add scoring + Redis write after snapshot dict**

In `_candle_logger_loop`, find the closing `}` of `self._mid_candle_snaps[_in_progress_key] = {` dict (ends at `"k5_candle_ts": _k5_candle_ts,` on line ~763). After the closing `}` and before the `logger.debug(...)` call, insert:

```python
                                _mid_candle_prob = None
                                if self._mid_candle_model is not None:
                                    try:
                                        _mid_candle_prob = self._mid_candle_model.predict(
                                            self._mid_candle_snaps[_in_progress_key]
                                        )["prob_up"]
                                    except Exception:
                                        pass
                                self._mid_candle_snaps[_in_progress_key]["mid_candle_prob"] = _mid_candle_prob
                                if _mid_candle_prob is not None:
                                    self._redis.set(
                                        "mid_candle:prob",
                                        json.dumps({"prob": _mid_candle_prob, "candle_ts": _in_progress_key}),
                                        ex=600,
                                    )
```

- [ ] **Step 2: Read `mid_candle_model_prob` at candle close**

Find the block that reads values from `_mid_snap` at candle close time (around line 795). After the existing `_mid_snap` reads (the last one is `k5_candle_ts = _mid_snap.get("k5_candle_ts") if _mid_snap else None` around line 817), add:

```python
                mid_candle_model_prob = _mid_snap.get("mid_candle_prob") if _mid_snap else None
```

- [ ] **Step 3: Add `mid_candle_model_prob` to the INSERT column list**

In the INSERT block (around line 837), find `col_names`:

```python
                col_names = (
                    ...
                    "spread_change, oi_delta_at_midcandle, k5_candle_ts, "
                    + ", ".join(cols)
                )
```

Change `"spread_change, oi_delta_at_midcandle, k5_candle_ts, "` to `"spread_change, oi_delta_at_midcandle, k5_candle_ts, mid_candle_model_prob, "`.

- [ ] **Step 4: Add `mid_candle_model_prob` to the VALUES list and update placeholder count**

In the VALUES list, find `k5_candle_ts,` (around line 890) and add `mid_candle_model_prob,` immediately after:

```python
                        k5_candle_ts,
                        mid_candle_model_prob,
                        *vals,
```

Update the placeholder count from `35` to `36`:

```python
                placeholders = ", ".join(["?"] * (36 + len(cols)))
```

- [ ] **Step 5: Verify parse**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -c "import main; print('OK')"
```

Expected: `OK`.

---

## Task 7: main.py — pass `current_candle_ts` to both `checklist.run()` calls

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Initialize `_open_dt` before the try block**

Find the try block that computes `_open_dt` (around line 1036). Add `_open_dt = None` immediately before the `try:`:

```python
        _open_dt = None
        try:
            _close_str = market.get("close_time", "")
            _close_dt = datetime.fromisoformat(_close_str.replace("Z", "+00:00"))
            _market_open_unix = _close_dt.timestamp() - 900.0
            _k15_post_open: int | None = (
                1 if cached["candle_ts"].timestamp() >= _market_open_unix else 0
            )
            _open_dt = _close_dt.replace(second=0, microsecond=0) - timedelta(seconds=900)
            self._candle_ticker_map[_open_dt.isoformat()] = ticker
        except Exception:
            _k15_post_open = None
```

- [ ] **Step 2: Derive `_current_candle_ts` after the try block**

Immediately after the try/except block (before `logger.debug(...)`), add:

```python
        _current_candle_ts: str | None = _open_dt.isoformat() if _open_dt is not None else None
```

- [ ] **Step 3: Pass `current_candle_ts` to the first `checklist.run()` call**

Find `result = self._checklist.run(` (around line 1105). Add `current_candle_ts=_current_candle_ts,` as the last keyword argument:

```python
        result = self._checklist.run(
            signal=signal,
            best_ask_cents=best_ask_cents,
            best_bid_cents=best_bid_cents,
            available_contracts=available_contracts,
            current_exposure=current_exposure,
            same_timeframe_open=same_timeframe_open,
            composite_price=composite_price,
            edge_above_threshold=edge_above_threshold,
            fresh_kalshi_mid=fresh_kalshi_mid1,
            is_drifting=self._drift_monitor.is_drifting(),
            direction_win_rate=dir_win_rate,
            is_bootstrap=is_bootstrap,
            current_candle_ts=_current_candle_ts,
        )
```

- [ ] **Step 4: Pass `current_candle_ts` to the second `checklist.run()` call**

Find `result2 = self._checklist.run(` (around line 1202). Add `current_candle_ts=_current_candle_ts,` as the last keyword argument (same pattern as above but using `fresh_kalshi_mid2`).

- [ ] **Step 5: Verify parse**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -c "import main; print('OK')"
```

Expected: `OK`.

---

## Task 8: Full test suite + final commit: Full test suite + final commit

- [ ] **Step 1: Run full test suite**

```bash
cd "/Users/ezrakornberg/Kronos V2" && python3 -m pytest --tb=short -q 2>&1 | tail -15
```

Expected: all tests pass (556+ passing, 0 failures).

- [ ] **Step 2: Commit main.py changes**

```bash
cd "/Users/ezrakornberg/Kronos V2" && git add main.py && git commit -m "$(cat <<'EOF'
feat: mid-candle model load + Gate 12/15 wiring in main.py

Loads MidCandleModel at startup, scores at mid-candle snapshot time,
writes to Redis with candle_ts, persists mid_candle_model_prob to DB,
passes current_candle_ts to both checklist.run() calls.
Inert until models/mid_candle.pkl exists.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify service restarts cleanly**

```bash
launchctl kickstart -k gui/$(id -u)/com.kronos.v2
```

Wait 10 seconds, then:

```bash
grep -E "MidCandleModel|Gate 15|mid_candle" ~/Library/Logs/kronos-v2/kronos-v2.log 2>/dev/null | tail -5
```

Expected: log line like `MidCandleModel file not found — model not loaded` OR if pkl exists: `MidCandleModel loaded from models/mid_candle.pkl`. No crash.
