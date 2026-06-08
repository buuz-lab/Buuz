"""
Live regime v2 performance dashboard.

Usage:
    python3 scripts/watch_regime.py [--db trades.db] [--n 50]
    watch -n 30 python3 scripts/watch_regime.py   # refresh every 30s
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime, timezone


def _bar(frac: float, width: int = 24) -> str:
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def load(db: str, n: int) -> dict:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("""
            SELECT candle_ts, regime_prob, shap_coherence, kalshi_open_mid, btc_direction
            FROM candle_features
            WHERE regime_prob IS NOT NULL AND btc_direction IS NOT NULL
              AND kalshi_open_mid IS NOT NULL
            ORDER BY candle_ts DESC LIMIT ?
        """, (n,)).fetchall()
        candles = list(reversed(rows))

        rejections = conn.execute("""
            SELECT timestamp, ticker, failed_gate, regime_prob, shap_coherence,
                   signal_prob, deepseek_regime, outcome
            FROM gate_rejections
            WHERE regime_prob IS NOT NULL
            ORDER BY timestamp DESC LIMIT 40
        """).fetchall()

        total_rows = conn.execute(
            "SELECT COUNT(*) FROM candle_features WHERE features_stale=0 AND atm_iv IS NOT NULL"
        ).fetchone()[0]

        warm_trained_rows = None
        try:
            import json
            from pathlib import Path
            m = json.loads(Path("models/regime_last_trained.json").read_text())
            warm_trained_rows = m.get("trained_at_rows")
        except Exception:
            pass

        return {
            "candles": candles,
            "rejections": rejections,
            "total_rows": total_rows,
            "warm_trained_rows": warm_trained_rows,
        }
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="trades.db")
    p.add_argument("--n",  type=int, default=50)
    args = p.parse_args()

    data     = load(args.db, args.n)
    candles  = data["candles"]
    rejs     = data["rejections"]
    total    = data["total_rows"]
    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_c      = len(candles)

    print(f"\n{'='*70}")
    print(f"  Regime v2  —  {now_str}")
    print(f"  {n_c} candle rows  |  {len(rejs)} gate rejections  |  {total} qualifying rows")
    print(f"{'='*70}")

    # ── N+1 accuracy ──────────────────────────────────────────────────────────
    print(f"\n── N+1 Candle Accuracy  {'─'*47}")
    if n_c < 2:
        print(f"  (need ≥2 rows to compare — have {n_c})")
    else:
        # Fair Kalshi benchmark: use kalshi_open_mid[N+1] (Kalshi at candle N+1 open —
        # same moment the one-shot regime fires) not kalshi_open_mid[N] which is
        # Kalshi 15 minutes BEFORE regime fires on a completely different candle.
        pairs = [(candles[i][0][:16], candles[i][1], candles[i+1][3], candles[i+1][4])
                 for i in range(n_c - 1)]
        n = len(pairs)

        rb_all = sum((p - d)**2 for _, p, k, d in pairs) / n
        kb_all = sum((k - d)**2 for _, p, k, d in pairs) / n
        acc_all = sum(1 for _, p, k, d in pairs if int(p >= 0.5) == d) / n
        adv_all = (kb_all - rb_all) / kb_all * 100 if kb_all else 0
        beat    = "✓ beating Kalshi" if rb_all < kb_all else "✗ behind Kalshi"

        print(f"  All-time  n={n:<3d}  regime={rb_all:.3f} ({acc_all:.0%})  "
              f"kalshi={kb_all:.3f}  adv={adv_all:+.1f}%  {beat}")

        if n >= 5:
            last = pairs[-5:]
            rb5  = sum((p - d)**2 for _, p, k, d in last) / 5
            kb5  = sum((k - d)**2 for _, p, k, d in last) / 5
            acc5 = sum(1 for _, p, k, d in last if int(p >= 0.5) == d) / 5
            adv5 = (kb5 - rb5) / kb5 * 100 if kb5 else 0
            arrow = "↑ improving" if rb5 < rb_all else "↓ declining"
            print(f"  Last-5    n=5    regime={rb5:.3f} ({acc5:.0%})  "
                  f"kalshi={kb5:.3f}  adv={adv5:+.1f}%  {arrow}")

        # Per-candle table — regime_prob[N] predicting direction[N+1]
        # K.Brier uses kalshi_open_mid[N+1] (same moment regime fires)
        offset = max(0, n - 10)
        print(f"\n  {'At close [N]':<16}  {'Pred':>5}  {'Dir[N+1]':>8}  Res  R.Brier  K.Brier(N+1)")
        print(f"  {'─'*16}  {'─'*5}  {'─'*8}  ───  ───────  ────────────")
        for i, (ts, p, k, d) in enumerate(pairs[-10:]):
            ok   = int(p >= 0.5) == d
            rb   = (p - d)**2
            kb   = (k - d)**2
            pred = f"{'↑' if p >= 0.5 else '↓'}{p:.2f}"
            nxt  = f"{'↑' if d else '↓'}"
            win  = "✓" if ok else "✗"
            star = " ★" if rb < kb else "  "
            print(f"  {candles[offset+i][0][:16]}  {pred:<5}  {nxt:<8}  {win}    {rb:.3f}    {kb:.3f}{star}")

        if n < 20:
            need_go  = max(0, 20 - n)
            need_cal = max(0, 10 - n)
            notes = []
            if need_cal > 0:
                notes.append(f"{need_cal} more for tier stats")
            if need_go > 0:
                notes.append(f"{need_go} more for go-live read")
            print(f"\n  Next: {' · '.join(notes)}")

    # ── Gate rejection record ─────────────────────────────────────────────────
    print(f"\n── Gate Rejections (same-candle record)  {'─'*30}")

    by_ticker: dict[str, list] = {}
    for r in rejs[:35]:
        t = r[1][-15:]
        by_ticker.setdefault(t, []).append(r)

    resolved = [(entries, next((e[7] for e in entries if e[7] is not None), None))
                for entries in by_ticker.values()]
    res_outcomes = [o for _, o in resolved if o is not None]
    wins  = sum(1 for o in res_outcomes if o == 1)
    total_res = len(res_outcomes)

    # Cache stability (post one-shot fix)
    cached_markets = sum(
        1 for entries, _ in resolved
        if len([e[3] for e in entries]) > 1 and
           max(e[3] for e in entries) - min(e[3] for e in entries) < 0.001
    )
    total_multi = sum(1 for entries, _ in resolved if len(entries) > 1)

    if total_res:
        print(f"  Resolved: {wins}/{total_res} wins ({wins/total_res:.0%})  "
              f"| Cache stable: {cached_markets}/{total_multi} multi-entry markets")
    else:
        print(f"  No resolved outcomes yet")

    print(f"\n  {'Time':<5}  {'Market':<15}  {'Regime':<13}  {'Coh':>4}  {'Gates':<8}  Outcome")
    for ticker, entries in list(by_ticker.items())[:8]:
        ts      = datetime.fromtimestamp(entries[0][0], tz=timezone.utc).strftime("%H:%M")
        probs   = [e[3] for e in entries]
        cohs    = [e[4] for e in entries if e[4] is not None]
        gates   = ",".join(str(e[2]) for e in entries)
        outcome = next((e[7] for e in entries if e[7] is not None), None)
        avg_coh = sum(cohs) / len(cohs) if cohs else 0
        result  = "WIN " if outcome == 1 else "LOSS" if outcome == 0 else "pending"

        stable = len(probs) > 1 and max(probs) - min(probs) < 0.001
        if len(probs) == 1:
            prob_str = f"{probs[0]:.3f} (1x)"
        elif stable:
            prob_str = f"{probs[0]:.3f} ×{len(probs)}✓"
        else:
            prob_str = f"{min(probs):.2f}→{max(probs):.2f} pre"

        print(f"  {ts:<5}  {ticker:<15}  {prob_str:<13}  {avg_coh:.2f}  {gates:<8}  {result}")

    # ── Progress ──────────────────────────────────────────────────────────────
    print(f"\n── Milestones  {'─'*55}")

    warm_trained = data["warm_trained_rows"] or 682
    warm_next    = warm_trained + 50
    warm_left    = max(0, warm_next - total)
    warm_pct     = min(1.0, (total - warm_trained) / 50) if warm_left > 0 else 1.0
    warm_status  = "FIRES NOW" if warm_left == 0 else f"{warm_left} rows  (~{warm_left//6}h)"
    print(f"  Warm-start  {_bar(warm_pct)} {warm_status}")

    n_c_total = n_c
    cal_pct   = min(1.0, n_c_total / 10)
    cal_left  = max(0, 10 - n_c_total)
    print(f"  Tier stats  {_bar(cal_pct)} {n_c_total}/10 candles  "
          f"({'ready' if cal_left == 0 else f'{cal_left} more'})")

    # Phase 3c: count unique resolved markets
    seen: set[str] = set()
    unique_res = 0
    for r in rejs:
        if r[7] is not None and r[1] not in seen:
            seen.add(r[1])
            unique_res += 1
    c3_pct = min(1.0, unique_res / 500)
    print(f"  Phase 3c    {_bar(c3_pct)} {unique_res}/500 unique markets (~Day 18-20)")

    print()


if __name__ == "__main__":
    main()
