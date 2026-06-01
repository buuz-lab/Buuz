# Session 23 — Claude Code Prompt

## Context

You are working on **Kronos V2** — a live BTC prediction-market trading system on Kalshi (KXBTC15M 15-min up/down markets). The project is at `~/Kronos V2`. **Read `handoff.md` in full before writing a single line of code.** The test suite is at 395 tests — do not break it.

Three targeted changes this session. All are backed by live data. Read the reasoning before implementing.

---

## Task 1: Gate 11 — Overconfidence guard

### Why

Post-May 26 data (all current gates live): YES fills at 30–45¢ where `kronos_calibrated > 0.75` have a **15.4% win rate on 13 trades** (-$19.69). This is the "maximum Kronos confidence + market strongly disagrees" pattern. When k_cal is 0.75–1.0 but the market prices YES below 45¢, the market has been right every time.

Gate 8 doesn't catch this: for a YES fill at 35¢, opposing margin = 0.15, which is below the 0.25 threshold. The calibrator won't fix it either — in the combined training data (trades + gate rejections), k15 >= 0.8 shows 54% y_up rate (gate rejections dilute the inversion), so the calibrator maps k_raw=1.0 to k_cal≈0.56, keeps direction YES, and still fires the trade at reduced size.

### What to build

In `btc_kalshi_system/execution/pretrade_checklist.py`, add Gate 11 **after** the `trade_price_cents` computation (after the YES/NO direction branch) and **before** Kelly runs. Exact insertion point: after the Gate 2a minimum price check (line ~73), before the `loss_streak` Redis read.

```python
# Gate 11 — Overconfidence guard
# Block YES trades where Kronos is at high confidence (k_cal > 0.75) but the
# market prices strongly against us (YES fill < 45¢). In this zone, the market's
# disagreement is informative: post-May-26 data shows 15% win rate on 13 trades.
# The calibrator compresses k_raw=1.0 to ~0.56 but keeps direction YES, so this
# gate is still needed after calibrator activates.
# Only applies to YES direction — NO direction at low prices has different dynamics.
_OVERCONFIDENCE_K_CAL_FLOOR = 0.75
_OVERCONFIDENCE_MAX_FILL_CENTS = 45
if (signal.direction == 1
        and signal.kronos_calibrated > _OVERCONFIDENCE_K_CAL_FLOOR
        and trade_price_cents < _OVERCONFIDENCE_MAX_FILL_CENTS):
    return fail(
        11,
        f"Overconfidence guard: k_cal={signal.kronos_calibrated:.2f} but "
        f"YES fill {trade_price_cents}¢ < {_OVERCONFIDENCE_MAX_FILL_CENTS}¢ "
        f"(market disagrees strongly; 15% historical win rate in this zone)",
    )
```

### Tests: `tests/execution/test_pretrade_checklist.py`

Add at minimum 4 tests:

1. `test_gate11_fires_high_kcal_low_fill_yes` — direction=YES, k_cal=0.85, fill=40¢ → fails gate 11
2. `test_gate11_does_not_fire_high_fill` — direction=YES, k_cal=0.85, fill=50¢ → passes gate 11
3. `test_gate11_does_not_fire_low_kcal` — direction=YES, k_cal=0.60, fill=35¢ → passes gate 11
4. `test_gate11_does_not_fire_no_direction` — direction=NO, k_cal=0.85, fill=35¢ → passes gate 11 (NO at 35¢ = YES at 65¢, different dynamics)

---

## Task 2: Reduce regime weight from 0.4 → 0.2

### Why

The regime model (v1, trained session 21) has a confirmed circular label problem: `direction == outcome` as the training label means its #1 feature (`kalshi_implied_prob` at 19% importance) is largely learning "when does Kalshi disagree with Kronos?" — not market microstructure.

Today's live evidence: regime model fired bearish (prob 0.17–0.19) on a bullish day with 76% win rate on Gate 2 shadow trades. More concretely, the 0.4 regime weight is **actively contaminating the fusion signal in ranging and high_uncertainty sessions**:

- k15_cal = 0.75 (bullish), regime_prob = 0.18 (bearish, wrong)
- Current: `combined = 0.6 × 0.75 + 0.4 × 0.18 = 0.522` → edge at 50¢ fill = 0.022 → Gate 5 blocks
- At 0.2 weight: `combined = 0.8 × 0.75 + 0.2 × 0.18 = 0.636` → edge = 0.136 → passes Gate 5

Gate 2 is shadow mode so the regime model doesn't block trades directly — but its weight in fusion means a bad regime signal causes Gate 5 and Gate 3 to block trades that the signal alone would have taken. This is the mechanism behind today's 39 Gate 2 shadow rejections at 76% win rate being stopped downstream.

0.4 was set in session 21 assuming the regime model was trustworthy. With the circular label confirmed, it's not. 0.2 reduces its influence while preserving the signal for the minority of cases where it's actually informative.

### What to change

In `btc_kalshi_system/signal/fusion.py`, change the module-level constants:

```python
# Before:
_KRONOS_WEIGHT = 0.6
_REGIME_WEIGHT = 0.4

# After:
_KRONOS_WEIGHT = 0.8
_REGIME_WEIGHT = 0.2
```

Add a comment on `_REGIME_WEIGHT` explaining why:
```python
# Reduced from 0.4 → 0.2 (session 23): regime model v1 has circular label
# (direction==outcome; kalshi_implied_prob is #1 feature at 19%). On bullish days,
# model fires bearish and contaminates fusion. Restore to 0.4 after regime v2 retrains
# with BTC-direction label. See handoff.md — Phase 1b.
```

### Tests to update

`tests/signal/test_fusion.py` has two tests that hardcode the old weights:

1. `test_combined_weighted_average` — docstring says `0.6 * kronos_cal + 0.4 * regime_prob`, hardcodes `expected = 0.6 * 0.70 + 0.4 * 0.80`. Update expected to `0.8 * 0.70 + 0.2 * 0.80` and update the docstring.

2. `test_combined_varies_with_regime_weight` — docstring says "Regime contributes 40%". Update to "Regime contributes 20%". The assertion logic (high regime_prob → higher combined) is still correct; only the docstring changes.

Scan for any other hardcoded `0.6`/`0.4` in fusion tests and update them. Use `grep -n "0\.6\|0\.4\|KRONOS_WEIGHT\|REGIME_WEIGHT" tests/signal/` to check.

---

---

## Task 3: Disagreement neutralization in fusion.py

### Why

After Task 2 reduces regime weight to 0.2, contamination persists on bad-regime days. Concrete math with today's live values (regime_prob = 0.18 bearish, k15_cal = 0.70 bullish):

- Task 2 only: `combined = 0.8×0.70 + 0.2×0.18 = 0.596` → edge at 50¢ fill = 0.096 → Gate 5 blocks
- Gate 5 requires combined ≥ 0.65. At w=0.2, k15 needs to be ≥ **0.75** to clear. Average k15 is 0.514.

Root cause: regime v1 has a circular label (`direction == outcome`). When regime and Kronos **disagree** directionally, the regime's deviation from 0.5 is largely noise — Gate 2 shadow data confirmed this: 39 disagreements at 76% win rate on 2026-05-30, regime wrong every time. The downward drag from a bearish regime_prob=0.18 reduces combined from 0.70 to 0.596 with no genuine predictive value.

**Disagreement neutralization:** when `regime_prob` and `kronos_cal` are on opposite sides of 0.5, use `regime_prob = 0.5` in the fusion formula. This zeroes the drag while keeping regime amplification intact on agreement days.

Effect at k15_cal=0.70, regime_prob=0.18:
- Task 2 alone: combined = 0.8×0.70 + 0.2×0.18 = **0.596** → Gate 5 blocks ✗
- Task 2 + neutralization: combined = 0.8×0.70 + 0.2×**0.50** = **0.66** → edge = 0.16 → passes ✓
- Minimum k15 to clear Gate 5 at 50¢: k15 ≥ **0.6875** (vs 0.75 with Task 2 only; vs 0.65 with no regime)

On agreement days (regime_prob=0.65, k15=0.70): no neutralization — combined = 0.8×0.70 + 0.2×0.65 = 0.69. Regime still amplifies.

Remove this logic after regime v2 deploys with `btc_direction` label (regime becomes genuinely independent — disagreements are informative again). See handoff.md — Phase 1b.

### What to change

In `btc_kalshi_system/signal/fusion.py`, replace the single-line `combined = ...` inside the `try:` block (after the Gate 2 disagreement logging block, before the regime shrinks) with:

```python
# Disagreement neutralization (session 23): when regime opposes Kronos direction,
# treat regime as neutral (0.5) to prevent downward drag on the combined signal.
# Regime v1 has circular label — disagreements are noise, not information. On
# agreement days regime contribution is fully preserved. Remove after regime v2
# deploys with btc_direction label. See handoff.md — Phase 1b.
_kronos_bullish = kronos_cal > 0.5
_regime_bullish = regime_prob > 0.5
if _kronos_bullish != _regime_bullish:
    _regime_in_fusion = 0.5
else:
    _regime_in_fusion = regime_prob
combined = _KRONOS_WEIGHT * kronos_cal + _REGIME_WEIGHT * _regime_in_fusion
```

The old line `combined = _KRONOS_WEIGHT * kronos_cal + _REGIME_WEIGHT * regime_prob` is replaced entirely.

### Tests to update

`tests/signal/test_fusion.py` has one existing test that uses a disagreeing regime and hardcodes the old combined value:

`test_gate2_shadow_mode_does_not_block` — uses `kronos_cal=0.70, regime_prob=0.30, regime_direction=0`. With neutralization, `regime_prob=0.30` (bearish) disagrees with `kronos_cal=0.70` (bullish), so `_regime_in_fusion=0.5`. Update:

```python
# Before:
expected = 0.8 * 0.70 + 0.2 * 0.30   # 0.62 — old behavior, regime dragged down
# After:
expected = 0.8 * 0.70 + 0.2 * 0.50   # 0.66 — regime neutralized on disagreement
```

Also update the inline comment from `"Combined blend still computed with both inputs since Gate 2 didn't block"` to reflect the neutralization.

Add 4 new tests:

1. `test_disagreement_neutralization_bullish_kronos_bearish_regime` — `kronos_cal=0.70, regime_prob=0.20` (disagree) → `combined = 0.8×0.70 + 0.2×0.50 = 0.66`, NOT `0.8×0.70 + 0.2×0.20 = 0.60`
2. `test_disagreement_neutralization_bearish_kronos_bullish_regime` — `kronos_cal=0.30, regime_prob=0.80` (disagree) → `combined = 0.8×0.30 + 0.2×0.50 = 0.34`, NOT `0.8×0.30 + 0.2×0.80 = 0.40`
3. `test_disagreement_neutralization_does_not_fire_on_agreement_bullish` — `kronos_cal=0.70, regime_prob=0.65` (both bullish) → `combined = 0.8×0.70 + 0.2×0.65 = 0.69` (no neutralization)
4. `test_disagreement_neutralization_does_not_fire_on_agreement_bearish` — `kronos_cal=0.35, regime_prob=0.20` (both bearish) → `combined = 0.8×0.35 + 0.2×0.20 = 0.32` (no neutralization)

Scan for any other tests that use opposing `kronos_cal`/`regime_prob` and hardcode a `combined` expectation. Use `grep -n "regime_prob.*0\.[0-3]\|0\.[0-3].*regime_prob" tests/signal/test_fusion.py` to catch them.

---

## After all three tasks

1. Run `python3 -m pytest tests/ -v` — all 403 (+ new Task 3 tests) must pass.
2. Add a session 23 Task 3 entry to `handoff.md` Current Progress section (table of files changed, what changed, why — follow the existing pattern exactly).

## Key constraints

- Gate 11 uses `signal.kronos_calibrated` — that's the k15-calibrated probability. With passthrough calibrator, this equals k_raw. After calibrator activates, it equals the compressed value. The gate fires on whatever the current calibrated value is — this is correct behavior.
- Gate 11 only applies to `direction == 1` (YES). Do not add NO direction logic — NO at 35¢ means buying down at 65¢, which is in the profitable price bucket and has different dynamics.
- The `_OVERCONFIDENCE_K_CAL_FLOOR` and `_OVERCONFIDENCE_MAX_FILL_CENTS` constants should be defined locally inside the `run()` method (same pattern as `_MIN_TRADE_PRICE_CENTS` on line 71), not at module level.
- `_KRONOS_WEIGHT` and `_REGIME_WEIGHT` must sum to 1.0 — verify after change.
- Do NOT change `REGIME_GATE2_ENFORCING` — Gate 2 stays shadow mode.
- Do NOT change Gate 5 thresholds — the regime weight reduction + neutralization naturally helps Gate 5 clearance; no further adjustment needed.
- Disagreement neutralization lives **inside the `try:` block** only (trained regime path). The `except NotTrainedError:` bootstrap path uses `combined = 0.5 + (kronos_cal - 0.5) * base_shrink` and does not use `regime_prob` at all — do NOT add neutralization logic there.
- `_kronos_bullish` and `_regime_bullish` are local variables inside the `try:` block. Do not define them at module level.
- The Gate 2 disagreement logging block (`logger.warning(...)`) still uses the raw `regime_prob` — do NOT change it to use `_regime_in_fusion`.
- `TradingSignal.regime_prob` stores the raw `regime_prob`, not `_regime_in_fusion`. This is intentional — the stored value should reflect what the model actually predicted for analysis/logging, not the neutralized version.
- Boundary case: `kronos_cal == 0.5` → `_kronos_bullish = False`; `regime_prob == 0.5` → `_regime_bullish = False` → they "agree" at boundary → no neutralization. This is correct behavior.
