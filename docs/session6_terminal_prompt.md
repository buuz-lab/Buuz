# Session 6 — Claude Code Terminal Prompt

Run this from the project root:
```
cd "/Users/ezrakornberg/Kronos V2" && claude
```
Then paste the prompt below.

---

## Prompt

You are implementing a Deribit options feed (session 6) for KronosV2, a live BTC prediction-market trading system. The system trades Kalshi 15-minute BTC up/down markets using a Kronos transformer + XGBoost regime classifier + DeepSeek gate.

Read `handoff.md` first for full context. Then read:
- `btc_kalshi_system/data/derivatives_feed.py` — the feed pattern to mirror exactly
- `btc_kalshi_system/signal/fusion.py` — where new features plug in (`_regime_features`, `TradingSignal`, `update_kalshi_mid`)
- `btc_kalshi_system/models/deepseek_parser.py` — where the DeepSeek prompt template lives
- `btc_kalshi_system/models/regime_model.py` — `_FEATURE_ORDER` (21 entries)
- `scripts/train_regime.py` — `_FEATURE_COLS`, `_EXTRA_FILTERS_20`, `_QUERY_TEMPLATE`
- `main.py` — `asyncio.gather`, `_get_market_context`, `_process_market`, `_record_trade_sqlite`, `_TRADES_COLUMN_MIGRATIONS`
- `tests/data/test_derivatives_feed.py` — test pattern to follow
- `config.py` — where constants live

Do NOT start implementing until you have read all of these files.

---

## What to build

A new isolated async feed, `btc_kalshi_system/data/deribit_options_feed.py`, that:
1. Polls Deribit's public options chain REST API every 5 minutes
2. Computes 5 metrics from the chain response
3. Writes them to Redis `options:features` (TTL 600s) + `options:features:lkg` (TTL 14400s = 4h)
4. Never crashes the main loop — all failures are logged and retried

Plus wiring changes to `fusion.py`, `deepseek_parser.py`, `regime_model.py`, `train_regime.py`, `main.py`, and new tests.

---

## The 6 new features (features 22–27)

### From `deribit_options_feed.py` → Redis `options:features` (4 keys):

**`atm_iv`** — At-the-money implied volatility for the nearest valid Deribit expiry. Annualised percentage as a float (e.g., 55.2 means 55.2%). Computation:
- Call `GET https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option`
- Response is `{"result": [...]}` where each element has: `instrument_name`, `underlying_price`, `mark_iv` (float, annualised %), `open_interest` (float, in BTC), `volume` (float, in BTC)
- Instrument name format: `BTC-{EXPIRY}-{STRIKE}-{C|P}` e.g. `BTC-27JUN25-100000-C`
- Parse expiry from instrument name (e.g. `27JUN25` → 27 Jun 2025 UTC midnight). Skip instruments with < 3 days to expiry (front-month theta spike).
- Select the nearest valid expiry. Filter to CALL instruments only (puts give the same ATM IV via put-call parity but calls tend to have better mark prices). Filter to instruments with `open_interest >= 10`. If no instruments pass, return None.
- Find the two call strikes that bracket `underlying_price` (one below, one above). Linearly interpolate `mark_iv` weighted by proximity to `underlying_price`. If only one side exists (spot above all strikes or below all), use the nearest available strike's `mark_iv` without interpolation.
- Fallback: if entire computation fails for any reason, write `None` / skip the field.

**`pcr_oi`** — Put/call ratio by open interest for the nearest valid expiry. Computation:
- Sum `open_interest` for all PUT instruments in the near expiry (OI >= 0, no minimum filter)
- Sum `open_interest` for all CALL instruments in the near expiry
- `pcr_oi = put_oi / call_oi` if call_oi > 0 else None
- Store as a float. Neutral = 1.0. > 1.0 means more put positioning (bearish hedge). < 1.0 means call-dominated (bullish positioning).
- Fallback: write 1.0 (not 0.0 — 0.0 would imply zero call OI, which is wrong).

**`term_structure_slope`** — Difference in ATM IV between near and far expiry, normalised. Computation:
- Compute ATM IV for the nearest valid expiry (near_iv) and the second-nearest valid expiry (far_iv), using the same interpolation method as `atm_iv`.
- `term_structure_slope = (far_iv - near_iv) / near_iv` if near_iv > 0 else 0.0
- Positive = far IV > near IV = contango (normal). Negative = near IV > far IV = backwardation (market pricing near-term stress).
- Fallback: 0.0.

**`skew_25d`** — 25-delta skew for the nearest valid expiry. Computation:
- Use `atm_iv` for the near expiry and `T` = days_to_expiry / 365.0.
- Approximate the 25Δ put strike: `put_strike ≈ underlying_price × (1 - 0.25 × (atm_iv/100) × sqrt(T))`
- Approximate the 25Δ call strike: `call_strike ≈ underlying_price × (1 + 0.25 × (atm_iv/100) × sqrt(T))`
- Find the listed PUT instrument with strike nearest to `put_strike` (within 5% of underlying). Read its `mark_iv`.
- Find the listed CALL instrument with strike nearest to `call_strike` (within 5% of underlying). Read its `mark_iv`.
- `skew_25d = put_iv - call_iv`
- Negative = puts more expensive = downside hedging. Positive = calls more expensive = upside demand.
- Fallback: 0.0.

### Derived in `_get_market_context()` — NOT written by the feed (1 key):

**`iv_rv_spread`** — ATM IV minus realised vol. Computation in `_get_market_context()` after merging both `regime:features` and `options:features`:
- `iv_rv_spread = ctx.get("atm_iv", 0.0) - ctx.get("brti_volatility_1h", 0.0)`
- Only compute if both values are present and non-zero. Otherwise 0.0.
- Positive = options pricing in more vol than is being realised (vol expensive). Negative = realised vol exceeds implied (vol cheap / market underpricing risk).

### Injected inline in `_process_market()` (1 key):

**`kalshi_spread_normalized`** — Kalshi bid-ask spread as a fraction of par. Computation in `_process_market()` after orderbook parse:
- `kalshi_spread_normalized = (best_ask_cents - best_bid_cents) / 100.0`
- Inject via `self._fusion.update_kalshi_spread(kalshi_spread_normalized)` immediately before the existing `self._fusion.update_kalshi_mid(mid_cents)` call.
- Add `update_kalshi_spread(self, spread: float) -> None` method to `SignalFusionEngine` that sets `self._market_context["kalshi_spread_normalized"] = spread`. Mirror the existing `update_kalshi_mid` exactly.

---

## Redis schema

```
options:features     → JSON dict with keys: atm_iv, pcr_oi, term_structure_slope, skew_25d
                       TTL: 600s (2× refresh interval, same as regime:features)
                       Written on every successful fetch.

options:features:lkg → same dict + "_lkg_written_at": time.time()
                       TTL: 14400s (4 hours — options data moves slower than perp futures)
                       Written on every successful fetch alongside options:features.
```

When `options:features` has expired but `:lkg` exists, `_get_market_context()` uses the LKG and adds `_deribit_lkg=True` to the context dict (same pattern as `_lkg=True` for `regime:features`). Log the age of the LKG in hours.

---

## Failure handling and reconnect

Mirror `derivatives_feed.py` exactly:

```python
async def run(self) -> None:
    while True:
        success = False
        try:
            features = await self._fetch_features()
            self._write_features(features)
            logger.info(f"DeribitOptionsFeed: wrote options:features — {features}")
            success = True
        except Exception as exc:
            logger.warning(f"DeribitOptionsFeed: fetch failed — {exc}")
        # On success: refresh 60s early (same headroom pattern as derivatives_feed)
        # On failure: wait full interval before retry
        await asyncio.sleep(_REFRESH_INTERVAL - 60 if success else _REFRESH_INTERVAL)
```

Deribit REST is stateless — there is no session to reconnect. Just retry after the sleep. Use a fresh `aiohttp.ClientSession` per fetch call (use `async with aiohttp.ClientSession() as session`).

The `_write_features` method must also write the LKG key:
```python
def _write_features(self, features: dict) -> None:
    serialized = json.dumps(features)
    self._redis.set("options:features", serialized, ex=_OPTIONS_TTL)
    lkg = dict(features)
    lkg["_lkg_written_at"] = time.time()
    self._redis.set("options:features:lkg", json.dumps(lkg), ex=_OPTIONS_LKG_TTL)
```

---

## Stale policy (STRICT)

Add a new `deribit_stale INTEGER DEFAULT 1` column to `trades.db`. ALL historical rows default to 1.

In `_regime_features()` in `fusion.py`, determine `deribit_stale` as follows:
```python
# deribit_stale=True when:
# (a) options:features key was absent (Deribit down and LKG also expired), OR
# (b) LKG was used (_deribit_lkg=True in ctx)
deribit_stale = (
    ctx.get("atm_iv") is None  # options data completely absent
    or ctx.get("_deribit_lkg", False)  # LKG fallback was used
)
```

Add `deribit_stale: bool = False` to `TradingSignal` dataclass (after `features_stale`).

In `_record_trade_sqlite()` in `main.py`, write `signal.deribit_stale` to the new column.

In `train_regime.py`, add `_EXTRA_FILTERS_27`:
```python
_EXTRA_FILTERS_27 = _EXTRA_FILTERS_20 + "\n  AND deribit_stale = 0\n  AND atm_iv IS NOT NULL"
```
The existing `_EXTRA_FILTERS_20` and 21-feature retrain path are UNCHANGED. The 27-feature path uses `_EXTRA_FILTERS_27` and `_FEATURE_COLS` extended to 27 entries.

---

## `_regime_features()` changes (fusion.py)

Add 6 new features at the bottom of the existing features dict, after `large_print_direction`. Use 0.0 as numeric fallbacks (not None) to keep XGBoost from blowing up:

```python
# --- Features 22–27: Deribit options + Kalshi spread ---
atm_iv = float(ctx.get("atm_iv") or 0.0)
iv_rv_spread = float(ctx.get("iv_rv_spread") or 0.0)
pcr_oi = float(ctx.get("pcr_oi") or 1.0)          # 1.0 not 0.0 — neutral ratio
term_structure_slope = float(ctx.get("term_structure_slope") or 0.0)
skew_25d = float(ctx.get("skew_25d") or 0.0)
kalshi_spread_normalized = float(self._market_context.get("kalshi_spread_normalized") or 0.0)
```

Also set `deribit_stale` per the stale policy above. `deribit_stale` is evaluated independently from the existing `stale` variable — do NOT OR them together. They are separate flags written to separate columns.

Return signature changes from `tuple[dict, bool]` to `tuple[dict, bool, bool]`: `(features, stale, deribit_stale)`. Update all callers (`get_signal()`).

---

## Feature order contract (3 files must be identical)

The existing test `test_feature_order` in the test suite enforces that `_FEATURE_ORDER` in `regime_model.py`, `_FEATURE_COLS` in `train_regime.py`, and the keys of the dict returned by `fusion._regime_features()` are all identical and in the same order.

Add these 6 keys in this exact order to all three locations:
```
"atm_iv", "iv_rv_spread", "pcr_oi", "term_structure_slope", "skew_25d", "kalshi_spread_normalized"
```
After `"large_print_direction"`.

Run `python3 -m pytest tests/ -k "feature_order"` to verify before proceeding.

---

## DeepSeek prompt update (deepseek_parser.py)

Add an OPTIONS MARKET section to `_PROMPT_TEMPLATE` between the DERIVATIVES section and the SENTIMENT & POSITIONING section:

```
OPTIONS MARKET (Deribit)
- ATM implied vol (near-term): {atm_iv}
- IV vs realized vol spread: {iv_rv_spread} (positive=options expensive vs realised, negative=vol cheap)
- Put/call ratio (OI): {pcr_oi} (>1.0=put-heavy positioning, <1.0=call-dominated)
- Vol term structure: {term_structure_slope} (positive=contango/normal, negative=backwardation/stress)
- 25Δ skew: {skew_25d} (negative=puts at premium=downside hedging, positive=call premium)
- Kalshi bid-ask spread: {kalshi_spread}¢ (wide=uncertain/thin market, narrow=consensus)
```

Add corresponding format variables in `_build_prompt()`. Use `"n/a"` when the field is None or missing. Format `atm_iv` as `f"{float(v):.1f}%"`, ratios to 2dp, spreads to 3dp.

---

## `_get_market_context()` changes (main.py)

After the existing `regime:derived_context` merge block, add:

```python
# Merge options features from DeribitOptionsFeed
try:
    opts_raw = r.get("options:features")
    if opts_raw:
        opts = json.loads(opts_raw)
        ctx.update({k: v for k, v in opts.items() if k not in ctx})
    else:
        opts_lkg_raw = r.get("options:features:lkg")
        if opts_lkg_raw:
            opts_lkg = json.loads(opts_lkg_raw)
            age_s = _time.time() - opts_lkg.pop("_lkg_written_at", _time.time())
            logger.warning(
                f"options:features expired — using LKG "
                f"({age_s / 3600:.1f}h old); row will be deribit_stale"
            )
            opts_lkg["_deribit_lkg"] = True
            ctx.update({k: v for k, v in opts_lkg.items() if k not in ctx})
except Exception:
    pass

# Derive iv_rv_spread from merged context (requires both sources to be present)
try:
    atm_iv = ctx.get("atm_iv")
    rv = ctx.get("brti_volatility_1h")
    if atm_iv is not None and rv is not None and rv > 0:
        ctx["iv_rv_spread"] = float(atm_iv) - float(rv)
except Exception:
    pass
```

---

## `main.py` wiring summary

1. Import `DeribitOptionsFeed` from `btc_kalshi_system.data.deribit_options_feed`.
2. In `KronosV2.run()`, instantiate `DeribitOptionsFeed()` and add `deribit_feed.run()` to the `asyncio.gather()` call alongside `deriv.run()`.
3. In `_process_market()`, add `self._fusion.update_kalshi_spread(kalshi_spread_normalized)` before `update_kalshi_mid`.
4. In `_record_trade_sqlite()`, add `deribit_stale` to the INSERT (write `1 if signal.deribit_stale else 0`).
5. Add to `_TRADES_COLUMN_MIGRATIONS`:
   ```python
   ("atm_iv",                    "REAL DEFAULT NULL"),
   ("iv_rv_spread",              "REAL DEFAULT NULL"),
   ("pcr_oi",                    "REAL DEFAULT NULL"),
   ("term_structure_slope",      "REAL DEFAULT NULL"),
   ("skew_25d",                  "REAL DEFAULT NULL"),
   ("kalshi_spread_normalized",  "REAL DEFAULT NULL"),
   ("deribit_stale",             "INTEGER DEFAULT 1"),
   ```

---

## SQL: trades.db new columns

All 7 new columns must appear in `_TRADES_COLUMN_MIGRATIONS` (idempotent ALTER TABLE — safe on existing DBs). The `_CREATE_TRADES_TABLE` does NOT need to be modified (new columns added via migration only). `deribit_stale DEFAULT 1` means all historical rows are correctly marked stale.

---

## Tests to write

Use TDD: write tests first, then implement.

### `tests/data/test_deribit_options_feed.py` (new file)

Use `fakeredis` and mock `aiohttp` (patch `aiohttp.ClientSession`). Follow the pattern in `test_derivatives_feed.py`.

Required tests:
1. `test_writes_options_features_to_redis` — mock a valid Deribit chain response with 2 expiries and several strikes bracketing a spot price. Assert `r.get("options:features")` contains all 4 keys (`atm_iv`, `pcr_oi`, `term_structure_slope`, `skew_25d`) after one successful fetch.
2. `test_writes_lkg_on_success` — assert `options:features:lkg` is written alongside the main key with a `_lkg_written_at` field.
3. `test_skips_expiry_under_3_days` — mock a chain where the only expiry is 1 day away. Assert `atm_iv` is None or feed uses the second expiry (if present) rather than the too-near one.
4. `test_pcr_oi_greater_than_one_when_puts_dominate` — mock chain with 200 BTC put OI and 100 BTC call OI. Assert `pcr_oi ≈ 2.0`.
5. `test_pcr_oi_fallback_is_one_not_zero` — mock chain with zero call OI. Assert `pcr_oi == 1.0`.
6. `test_term_structure_slope_positive_in_contango` — mock far IV > near IV. Assert `term_structure_slope > 0`.
7. `test_fetch_failure_does_not_write` — mock aiohttp to raise `aiohttp.ClientError`. Assert `r.get("options:features")` is None (no write). Assert LKG from a prior successful write is NOT overwritten.
8. `test_atm_iv_interpolation` — two bracketing strikes at spot ± 1000. Assert interpolated IV lies strictly between the two strikes' IVs.
9. `test_run_retries_after_failure` — patch `_fetch_features` to raise on first call, succeed on second. Assert `_write_features` is called exactly once after two cycles.

### `tests/signal/test_fusion_deribit_features.py` (new file or extend `test_fusion.py`)

1. `test_regime_features_includes_all_27_keys` — assert all 27 feature keys present in `_regime_features()` output.
2. `test_deribit_stale_true_when_options_features_absent` — context dict with no `atm_iv`. Assert `deribit_stale=True`.
3. `test_deribit_stale_true_when_lkg_used` — context dict with `_deribit_lkg=True`. Assert `deribit_stale=True`.
4. `test_deribit_stale_false_when_options_fresh` — context dict with valid `atm_iv`. Assert `deribit_stale=False`.
5. `test_iv_rv_spread_in_context` — context dict has both `atm_iv=60.0` and `brti_volatility_1h=0.01` (realised vol in CV units). Verify `iv_rv_spread` is computed and written to context.
6. `test_kalshi_spread_in_regime_features` — call `update_kalshi_spread(0.05)` then `_regime_features()`. Assert `kalshi_spread_normalized == 0.05` in the output dict.
7. `test_pcr_oi_default_is_one` — context dict missing `pcr_oi`. Assert `pcr_oi == 1.0` in features.

### Existing tests to extend:

- `test_feature_order` (wherever it lives) — add the 6 new keys in the correct position. Run this early to catch any mismatch.

---

## Invariants — do NOT break these

- `_FEATURE_COLS_LEGACY` in `train_regime.py` stays at exactly 6 entries. Do not touch it.
- `features_stale` and `deribit_stale` are separate flags. Never combine them. `features_stale` guards the 21-feature retrain; `deribit_stale` guards the additional 6 Deribit features.
- Do not add `_deribit_lkg`, `_lkg_written_at`, or any sentinel key to the features dict — these are internal context markers only. They must be stripped before XGBoost sees the data (they already are, since `_regime_features()` only reads named keys).
- The existing `update_kalshi_mid()` call in `_process_market()` stays unchanged. `update_kalshi_spread()` is a new method; do not modify or replace `update_kalshi_mid`.
- `pcr_oi` fallback is `1.0`, not `0.0`. This is intentional.
- Deribit IV is annualised percentage (e.g., 55.2). Store it as-is. Do not convert to decimal.
- The Deribit public API URL is `https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC&kind=option`. No API key required.

---

## Definition of done

1. `python3 -m pytest` — all existing tests pass + new tests pass
2. `python3 -m pytest tests/ -k "feature_order"` — passes with 27 features
3. `python3 main.py` starts without error; within 5 minutes `redis-cli get options:features` returns a valid JSON dict with all 4 keys
4. `redis-cli ttl options:features` returns 400–600
5. `redis-cli get options:features:lkg` is also populated
6. After one trade cycle, `trades.db` has `atm_iv`, `pcr_oi`, `term_structure_slope`, `skew_25d`, `kalshi_spread_normalized`, `iv_rv_spread`, and `deribit_stale` columns
7. `deribit_stale = 1` for all rows until fresh Deribit data flows through

---

## Final step

After all tests pass and the feed is verified running, update `handoff.md`:
- Change session 6 status from "Planned" to "Complete"
- Update the "Current Progress" timestamp
- Add the new files to the "Files Touched This Session" table
- Note the `deribit_stale=0` row accumulation start date
- Commit with message: `feat: Deribit options feed — 6 new regime features (22–27)`
