# Gate 15 + Mid-Candle Integration Design — 2026-06-08

## Goal

Wire the existing `MidCandleModel` infrastructure into the live trading loop. Three improvements:
1. Prevent a latent cross-candle Redis contamination bug in Gate 15
2. Add Gate 15 direction filter (mid-candle model score) with candle-ts validation
3. Restructure Gate 12 to allow a 40-60% entry window when the model is loaded

All changes are inert until `models/mid_candle.pkl` exists — no behavioral change today.

---

## Files Changed

| File | What changes |
|---|---|
| `main.py` | Model load in `__init__`, Redis write in `_candle_logger_loop`, schema column, `current_candle_ts` passed to both `checklist.run()` calls |
| `btc_kalshi_system/execution/pretrade_checklist.py` | Gate 12 restructured, Gate 15 added, `_mid_candle_model_loaded` attribute, `current_candle_ts` param |
| `tests/execution/test_pretrade_checklist.py` | 4 new tests for Gate 12 mid-window and Gate 15 |

---

## main.py Changes

### 1. `__init__` — load model, set flag

After `self._checklist = PreTradeChecklist(self._kelly)`:

```python
try:
    self._mid_candle_model = MidCandleModel.load(config.MID_CANDLE_MODEL_PATH)
    self._checklist._mid_candle_model_loaded = True
    logger.info(f"MidCandleModel loaded from {config.MID_CANDLE_MODEL_PATH}")
except FileNotFoundError:
    self._mid_candle_model = None
```

Import: `from btc_kalshi_system.models.mid_candle_model import MidCandleModel`

### 2. Schema — `_CANDLE_FEATURES_COLUMN_MIGRATIONS`

Add: `("mid_candle_model_prob", "REAL DEFAULT NULL")`

### 3. `_candle_logger_loop` — score and write to Redis

Immediately after `self._mid_candle_snaps[_in_progress_key] = {...}`:

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

Also add `mid_candle_model_prob` to the candle_features INSERT (read from `_mid_snap.get("mid_candle_prob")`).

### 4. Both `checklist.run()` calls — pass `current_candle_ts`

`_open_dt` is already computed at line 1044. Pass `current_candle_ts=_open_dt.isoformat()` to both the first and second `self._checklist.run()` calls.

---

## pretrade_checklist.py Changes

### 1. `__init__`

Add `self._mid_candle_model_loaded: bool = False` (main.py sets it to `True` after successful model load).

### 2. `run()` signature

Add `current_candle_ts: str | None = None` parameter.

### 3. Gate 12 — restructured for mid-window bypass

Replace the current flat floor/cap checks with:

```python
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

### 4. Gate 15 — placed immediately after Gate 12

```python
# Gate 15 — Mid-candle model direction filter (40-60% window only)
# Only fires when candle is in mid-window and model is loaded. Skips silently if
# Redis score is from a different candle (cross-candle contamination guard).
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

---

## Tests

Five new tests in `tests/execution/test_pretrade_checklist.py`:

```python
def test_gate12_mid_window_no_model_blocks(checklist):
    """Progress 50% but model not loaded → Gate 12 blocks."""
    signal = make_signal(regime_features={"candle_progress": 0.50})
    r = checklist.run(**base_kwargs(signal))
    assert r.failed_gate == 12
    assert "not loaded" in r.failed_reason

def test_gate12_mid_window_with_model_passes(checklist):
    """Progress 50% and model loaded → Gate 12 passes (falls through to Gate 15)."""
    checklist._mid_candle_model_loaded = True
    signal = make_signal(regime_features={"candle_progress": 0.50})
    r = checklist.run(**base_kwargs(signal))
    assert r.failed_gate != 12

def test_gate15_stale_candle_ts_skips(checklist):
    """Gate 15 silently skips when Redis candle_ts doesn't match current candle."""
    checklist._mid_candle_model_loaded = True
    checklist._redis.get.return_value = json.dumps(
        {"prob": 0.20, "candle_ts": "2026-01-01T00:00:00+00:00"}
    ).encode()
    signal = make_signal(regime_features={"candle_progress": 0.50})
    r = checklist.run(**base_kwargs(signal), current_candle_ts="2026-01-01T00:15:00+00:00")
    assert r.passed  # stale ts → gate skipped → trade allowed

def test_gate15_bearish_blocks_yes(checklist):
    """Gate 15 blocks YES when mid-candle model is bearish (prob < 0.38)."""
    checklist._mid_candle_model_loaded = True
    current_ts = "2026-01-01T00:15:00+00:00"
    checklist._redis.get.return_value = json.dumps(
        {"prob": 0.25, "candle_ts": current_ts}
    ).encode()
    signal = make_signal(direction=1, regime_features={"candle_progress": 0.50})
    r = checklist.run(**base_kwargs(signal), current_candle_ts=current_ts)
    assert r.failed_gate == 15
    assert "bearish" in r.failed_reason

def test_gate15_bullish_blocks_no(checklist):
    """Gate 15 blocks NO when mid-candle model is bullish (prob > 0.62)."""
    checklist._mid_candle_model_loaded = True
    current_ts = "2026-01-01T00:15:00+00:00"
    checklist._redis.get.return_value = json.dumps(
        {"prob": 0.75, "candle_ts": current_ts}
    ).encode()
    signal = make_signal(direction=0, regime_features={"candle_progress": 0.50})
    r = checklist.run(**base_kwargs(signal), current_candle_ts=current_ts)
    assert r.failed_gate == 15
    assert "bullish" in r.failed_reason
```

---

## Safety Invariants

- **No behavioral change today**: `_mid_candle_model_loaded = False` by default. Gate 12 mid-window blocks (same as the old cap check). Gate 15 never runs. Normal early-entry flow is identical.
- **Activation**: Place `models/mid_candle.pkl` and restart → model loads, flag flips True, both gates activate.
- **Cross-candle guard**: Gate 15 only uses a Redis score whose `candle_ts` matches the current candle's open ISO string. Stale scores are silently skipped, never block.
- **`_open_dt`** in main.py is already computed before both `checklist.run()` calls — no new Redis reads or API calls at gate time.
