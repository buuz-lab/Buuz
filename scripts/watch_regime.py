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
        candles = conn.execute("""
            SELECT candle_ts, regime_prob, shap_coherence, kalshi_open_mid, btc_direction
            FROM candle_features
            WHERE regime_prob IS NOT NULL AND btc_direction IS NOT NULL
              AND kalshi_open_mid IS NOT NULL
            ORDER BY candle_ts DESC LIMIT ?
        """, (n,)).fetchall()

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
    print("\n── Rolling Brier vs Kalshi " + "─" * 44)
    if not candles:
        print("  (no regime_prob candle rows yet — accumulating)")
        return

    probs   = [r[1] for r in candles]
    kprobs  = [r[3] for r in candles]
    dirs    = [r[4] for r in candles]

    regime_brier  = sum((p - d) ** 2 for p, d in zip(probs, dirs)) / len(probs)
    kalshi_brier  = sum((k - d) ** 2 for k, d in zip(kprobs, dirs)) / len(probs)
    regime_acc    = sum(1 for p, d in zip(probs, dirs) if int(p >= 0.5) == d) / len(probs)
    kalshi_acc    = sum(1 for k, d in zip(kprobs, dirs) if int(k >= 0.5) == d) / len(probs)
    adv = (kalshi_brier - regime_brier) / kalshi_brier * 100 if kalshi_brier > 0 else 0

    print(f"  n={len(candles):<4d}  "
          f"regime Brier={regime_brier:.4f} ({regime_acc:.0%})  "
          f"kalshi Brier={kalshi_brier:.4f} ({kalshi_acc:.0%})  "
          f"adv={adv:+.1f}%  {_brier_color(regime_brier)}")

    if len(candles) < 10:
        print(f"  ⚠  {10 - len(candles)} more candles needed for reliable comparison")


def section_calibration(candles: list) -> None:
    print("\n── Calibration by Confidence Tier " + "─" * 37)
    if len(candles) < 3:
        print("  (accumulating — need 10+ per tier for stats)")
        return

    tiers = [
        ("Low  |p-0.5|<0.10", lambda p: abs(p - 0.5) < 0.10),
        ("Med  0.10–0.20",     lambda p: 0.10 <= abs(p - 0.5) < 0.20),
        ("High |p-0.5|>0.20", lambda p: abs(p - 0.5) >= 0.20),
    ]
    for name, fn in tiers:
        subset = [(r[1], r[3], r[2]) for r in candles if fn(r[1])]
        n = len(subset)
        if n == 0:
            print(f"  {name:<20s}  n=0")
            continue
        ps, ds, chs = zip(*subset)
        brier  = sum((p - d) ** 2 for p, d in zip(ps, ds)) / n
        acc    = sum(1 for p, d in zip(ps, ds) if int(p >= 0.5) == d) / n
        avg_coh = sum(c for c in chs if c is not None) / max(1, sum(1 for c in chs if c is not None))
        stats = f"Brier={brier:.3f}  acc={acc:.0%}  coh={avg_coh:.3f}" if n >= 10 else f"n={n} (accumulating)"
        bar = _bar(acc) if n >= 10 else ""
        print(f"  {name:<20s}  n={n:<3d}  {stats}  {bar}")


def section_shap(candles: list) -> None:
    print("\n── SHAP Coherence vs Outcome " + "─" * 42)
    resolved = [(r[1], r[2], r[4]) for r in candles if r[2] is not None]
    if len(resolved) < 3:
        print("  (accumulating)")
        return

    # Does higher coherence correlate with correctness?
    correct   = [c for p, c, d in resolved if int(p >= 0.5) == d]
    incorrect = [c for p, c, d in resolved if int(p >= 0.5) != d]
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
    print("\n── Recent Signals (gate rejections) " + "─" * 35)
    if not rejections:
        print("  (none yet)")
        return

    # Group by ticker to show per-market story
    by_ticker: dict[str, list] = {}
    for r in rejections[:20]:
        t = r[1][-15:] if len(r[1]) > 15 else r[1]
        by_ticker.setdefault(t, []).append(r)

    for ticker, entries in list(by_ticker.items())[:6]:
        ts_latest = datetime.fromtimestamp(entries[0][0], tz=timezone.utc).strftime("%H:%M")
        probs = [e[3] for e in entries]
        cohs  = [e[4] for e in entries if e[4] is not None]
        gates = [str(e[2]) for e in entries]
        outcome = next((e[7] for e in entries if e[7] is not None), None)
        outcome_str = f"→ {'WIN' if outcome == 1 else 'LOSS'}" if outcome is not None else "→ pending"
        avg_prob = sum(probs) / len(probs)
        avg_coh  = sum(cohs) / len(cohs) if cohs else 0
        prob_str = f"[{min(probs):.2f}→{max(probs):.2f}]" if len(probs) > 1 else f"{probs[0]:.3f}"
        print(f"  {ts_latest}  {ticker}  prob={prob_str}  coh={avg_coh:.3f}  "
              f"gates={','.join(gates)}  {outcome_str}")


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
