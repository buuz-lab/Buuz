"""
Live regime v2 performance dashboard.

Shows rolling Brier, calibration by confidence tier, gate rejection breakdown,
SHAP coherence, and per-market signal history.

Usage:
    python3 scripts/watch_regime.py [--db trades.db] [--n 50]
    watch -n 30 python3 scripts/watch_regime.py   # refresh every 30s
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# ── Formatting ────────────────────────────────────────────────────────────────

def _bar(frac: float, width: int = 20) -> str:
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)

def _brier_color(b: float) -> str:
    if b < 0.22:   return "★ BEATS KALSHI"
    if b < 0.25:   return "↑ near coin-flip"
    return         "  above coin-flip"

def _dir(d: int | None) -> str:
    if d is None: return "?"
    return "UP  ✓" if d == 1 else "DOWN✓"


# ── Data loading ──────────────────────────────────────────────────────────────

def load(db: str, n: int) -> dict:
    conn = sqlite3.connect(db)
    try:
        # Candle-level rows with regime_prob
        # DESC + reverse: get most-recent n rows but in chronological order for lag pairs
        rows = conn.execute("""
            SELECT candle_ts, regime_prob, shap_coherence, kalshi_open_mid, btc_direction
            FROM candle_features
            WHERE regime_prob IS NOT NULL AND btc_direction IS NOT NULL
              AND kalshi_open_mid IS NOT NULL
            ORDER BY candle_ts DESC LIMIT ?
        """, (n,)).fetchall()
        candles = list(reversed(rows))  # oldest→newest so pairs[i] predicts pairs[i+1]

        # Gate rejections with regime_prob (most recent first)
        rejections = conn.execute("""
            SELECT timestamp, ticker, failed_gate, regime_prob, shap_coherence,
                   signal_prob, deepseek_regime, outcome
            FROM gate_rejections
            WHERE regime_prob IS NOT NULL
            ORDER BY timestamp DESC LIMIT 30
        """).fetchall()

        # Total qualifying rows for row-trigger progress
        total_rows = conn.execute(
            "SELECT COUNT(*) FROM candle_features WHERE features_stale=0 AND atm_iv IS NOT NULL"
        ).fetchone()[0]

        return {"candles": candles, "rejections": rejections, "total_rows": total_rows}
    finally:
        conn.close()


# ── Display sections ──────────────────────────────────────────────────────────

def section_overview(data: dict, n: int) -> None:
    candles = data["candles"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_candles = len(candles)
    n_rej = len(data["rejections"])

    print(f"\n{'='*70}")
    print(f"  Regime v2 Live Dashboard  —  {now}")
    print(f"  {n_candles} candle rows  |  {n_rej} gate rejections  |  "
          f"{data['total_rows']} total qualifying rows")
    print(f"{'='*70}")


def section_brier(candles: list) -> None:
    # regime_prob[N] predicts direction[N+1] (1-candle lag = training objective).
    print("\n── N+1 Candle Accuracy (close-lag, how model was trained) " + "─" * 14)
    if len(candles) < 2:
        print(f"  (need ≥2 rows — have {len(candles)})")
        return

    pairs = [(candles[i][1], candles[i][3], candles[i+1][4]) for i in range(len(candles) - 1)]
    n = len(pairs)

    regime_brier = sum((p - d) ** 2 for p, k, d in pairs) / n
    kalshi_brier = sum((k - d) ** 2 for p, k, d in pairs) / n
    regime_acc   = sum(1 for p, k, d in pairs if int(p >= 0.5) == d) / n
    kalshi_acc   = sum(1 for p, k, d in pairs if int(k >= 0.5) == d) / n
    adv = (kalshi_brier - regime_brier) / kalshi_brier * 100 if kalshi_brier > 0 else 0
    beating = regime_brier < kalshi_brier

    print(f"  All-time  n={n:<3d}  "
          f"regime={regime_brier:.4f} ({regime_acc:.0%})  "
          f"kalshi={kalshi_brier:.4f} ({kalshi_acc:.0%})  "
          f"adv={adv:+.1f}%  {'✓ beating' if beating else '✗ behind'}")

    # Last-5 rolling window for trend
    if n >= 5:
        last5 = pairs[-5:]
        rb5 = sum((p - d) ** 2 for p, k, d in last5) / 5
        kb5 = sum((k - d) ** 2 for p, k, d in last5) / 5
        acc5 = sum(1 for p, k, d in last5 if int(p >= 0.5) == d) / 5
        adv5 = (kb5 - rb5) / kb5 * 100 if kb5 > 0 else 0
        trend = "↑" if rb5 < regime_brier else "↓"
        print(f"  Last-5    n=5    "
              f"regime={rb5:.4f} ({acc5:.0%})  "
              f"kalshi={kb5:.4f}  "
              f"adv={adv5:+.1f}%  {trend} vs all-time")

    # Per-candle row
    print(f"\n  {'Candle':<16}  {'prob':>5}  {'→next':>5}  {'result':>6}  {'R.Brier':>7}  {'K.Brier':>7}")
    for i, (p, k, d) in enumerate(pairs[-10:]):
        ok = int(p >= 0.5) == d
        rb = (p - d) ** 2
        kb = (k - d) ** 2
        print(f"  {candles[i][0][:16]}  {p:.2f}   {'↑' if d else '↓'}     {'✓' if ok else '✗'}      {rb:.3f}    {kb:.3f}")

    if n < 10:
        print(f"\n  ⚠  {10 - n} more candles for reliable stats")


def section_calibration(candles: list) -> None:
    print("\n── Calibration by Confidence Tier (1-candle lag) " + "─" * 22)
    if len(candles) < 3:
        print("  (accumulating — need 10+ per tier for stats)")
        return

    # 1-candle lag pairs
    pairs = [(candles[i][1], candles[i][3], candles[i+1][4], candles[i][2])
             for i in range(len(candles) - 1)]  # (prob, kalshi, next_dir, coh)

    tiers = [
        ("Low  |p-0.5|<0.10", lambda p: abs(p - 0.5) < 0.10),
        ("Med  0.10–0.20",     lambda p: 0.10 <= abs(p - 0.5) < 0.20),
        ("High |p-0.5|>0.20", lambda p: abs(p - 0.5) >= 0.20),
    ]
    for name, fn in tiers:
        subset = [(p, k, d, c) for p, k, d, c in pairs if fn(p)]
        n = len(subset)
        if n == 0:
            print(f"  {name:<20s}  n=0")
            continue
        ps  = [x[0] for x in subset]
        ks  = [x[1] for x in subset]
        ds  = [x[2] for x in subset]
        chs = [x[3] for x in subset if x[3] is not None]
        brier  = sum((p - d) ** 2 for p, d in zip(ps, ds)) / n
        acc    = sum(1 for p, d in zip(ps, ds) if int(p >= 0.5) == d) / n
        avg_coh = sum(chs) / len(chs) if chs else 0
        stats = f"Brier={brier:.3f}  acc={acc:.0%}  coh={avg_coh:.3f}" if n >= 10 else f"n={n} (accumulating)"
        bar = _bar(acc) if n >= 10 else ""
        print(f"  {name:<20s}  n={n:<3d}  {stats}  {bar}")


def section_shap(candles: list) -> None:
    print("\n── SHAP Coherence vs Outcome " + "─" * 42)
    # 1-candle lag: coherence at candle N vs correctness predicting candle N+1
    pairs = [(candles[i][1], candles[i][2], candles[i+1][4])
             for i in range(len(candles) - 1) if candles[i][2] is not None]
    if len(pairs) < 3:
        print("  (accumulating)")
        return

    correct   = [c for p, c, d in pairs if int(p >= 0.5) == d]
    incorrect = [c for p, c, d in pairs if int(p >= 0.5) != d]
    avg_c_ok  = sum(correct)   / len(correct)   if correct   else 0
    avg_c_no  = sum(incorrect) / len(incorrect) if incorrect else 0
    n_ok, n_no = len(correct), len(incorrect)

    print(f"  Correct predictions    n={n_ok:<3d}  avg_coherence={avg_c_ok:.3f}  {_bar(avg_c_ok)}")
    print(f"  Incorrect predictions  n={n_no:<3d}  avg_coherence={avg_c_no:.3f}  {_bar(avg_c_no)}")
    if n_ok >= 3 and n_no >= 3:
        delta = avg_c_ok - avg_c_no
        symbol = "✓" if delta > 0 else "✗"
        print(f"  {symbol} Coherence delta: {delta:+.3f}  "
              f"({'correct calls more coherent' if delta > 0 else 'no coherence advantage yet'})")


def section_recent_signals(rejections: list) -> None:
    # One-shot cache: since the fix, regime_prob should be identical across
    # all gate rejections within the same market. "stable" = cache working.
    print("\n── Recent Signals (gate rejections) " + "─" * 35)
    if not rejections:
        print("  (none yet)")
        return

    by_ticker: dict[str, list] = {}
    for r in rejections[:24]:
        t = r[1][-15:] if len(r[1]) > 15 else r[1]
        by_ticker.setdefault(t, []).append(r)

    for ticker, entries in list(by_ticker.items())[:7]:
        ts = datetime.fromtimestamp(entries[0][0], tz=timezone.utc).strftime("%H:%M")
        probs   = [e[3] for e in entries]
        cohs    = [e[4] for e in entries if e[4] is not None]
        gates   = [str(e[2]) for e in entries]
        outcome = next((e[7] for e in entries if e[7] is not None), None)
        result  = f"→ {'WIN' if outcome == 1 else 'LOSS'}" if outcome is not None else "→ pending"
        avg_coh = sum(cohs) / len(cohs) if cohs else 0

        stable = max(probs) - min(probs) < 0.001
        if len(probs) == 1:
            prob_str  = f"{probs[0]:.3f}"
            cache_tag = ""
        elif stable:
            prob_str  = f"{probs[0]:.3f} ×{len(probs)}"
            cache_tag = " [cache✓]"
        else:
            prob_str  = f"[{min(probs):.2f}→{max(probs):.2f}]"
            cache_tag = " [pre-fix]"

        print(f"  {ts}  {ticker}  prob={prob_str}  coh={avg_coh:.3f}  "
              f"gates={','.join(gates)}  {result}{cache_tag}")


def section_row_progress(data: dict) -> None:
    print("\n── Training Data Progress " + "─" * 45)
    n_candles = len(data["candles"])
    total     = data["total_rows"]

    # Regime warm-start trigger
    warm_trigger = 732  # 682 + 50
    warm_remaining = max(0, warm_trigger - total)
    warm_pct = min(1.0, total / warm_trigger)
    print(f"  Warm-start retrain   : {total}/{warm_trigger}  {_bar(warm_pct, 30)}  "
          f"{'READY' if warm_remaining == 0 else f'{warm_remaining} rows away'}")

    # Go-live check
    live_trigger = 20
    live_pct = min(1.0, n_candles / live_trigger)
    print(f"  Go-live check (≥20)  : {n_candles}/{live_trigger}  {_bar(live_pct, 30)}  "
          f"{'READY' if n_candles >= live_trigger else f'{live_trigger - n_candles} rows away'}")

    # Phase 3c calibrator
    c3_rej = sum(1 for r in data["rejections"] if r[7] is not None)
    c3_trigger = 500
    c3_pct = min(1.0, c3_rej / c3_trigger)
    print(f"  Phase 3c calibrator  : ~{c3_rej}/{c3_trigger}  {_bar(c3_pct, 30)}  "
          f"{c3_trigger - c3_rej} resolved entries away")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="trades.db")
    p.add_argument("--n",  type=int, default=50,
                   help="Rolling window of candle rows to analyse (default: 50)")
    args = p.parse_args()

    data = load(args.db, args.n)
    section_overview(data, args.n)
    section_brier(data["candles"])
    section_calibration(data["candles"])
    section_shap(data["candles"])
    section_recent_signals(data["rejections"])
    section_row_progress(data)
    print()


if __name__ == "__main__":
    main()
