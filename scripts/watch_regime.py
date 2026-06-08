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
from pathlib import Path


def _bar(frac: float, width: int = 22) -> str:
    filled = int(round(min(frac, 1.0) * width))
    return "█" * filled + "░" * (width - filled)


def load(db: str, n: int) -> dict:
    conn = sqlite3.connect(db)
    try:
        rows = conn.execute("""
            SELECT candle_ts, regime_prob, shap_coherence, kalshi_open_mid, btc_direction,
                   kalshi_early_mid, kalshi_early_progress
            FROM candle_features
            WHERE regime_prob IS NOT NULL AND btc_direction IS NOT NULL
              AND kalshi_open_mid IS NOT NULL
            ORDER BY candle_ts DESC LIMIT ?
        """, (n,)).fetchall()
        candles = list(reversed(rows))   # oldest → newest for pair maths

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
            m = json.loads(Path("models/regime_last_trained.json").read_text())
            warm_trained_rows = m.get("trained_at_rows")
        except Exception:
            pass

        # Current candle's regime read (from latest gate rejection this candle)
        now_utc = datetime.now(timezone.utc)
        cur_min = (now_utc.minute // 15) * 15
        candle_open = now_utc.replace(minute=cur_min, second=0, microsecond=0)
        elapsed_s   = (now_utc - candle_open).total_seconds()
        progress    = elapsed_s / 900.0

        # Latest gate rejection that fired during the current candle open
        cur_regime = conn.execute("""
            SELECT regime_prob, shap_coherence, failed_gate
            FROM gate_rejections
            WHERE regime_prob IS NOT NULL
              AND timestamp >= ?
            ORDER BY timestamp DESC LIMIT 1
        """, (candle_open.timestamp(),)).fetchone()

        return {
            "candles": candles,
            "rejections": rejections,
            "total_rows": total_rows,
            "warm_trained_rows": warm_trained_rows,
            "candle_open": candle_open,
            "elapsed_s": elapsed_s,
            "progress": progress,
            "cur_regime": cur_regime,  # (regime_prob, shap_coh, gate) or None
        }
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="trades.db")
    p.add_argument("--n",  type=int, default=50)
    args = p.parse_args()

    data    = load(args.db, args.n)
    candles = data["candles"]
    rejs    = data["rejections"]
    total   = data["total_rows"]
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_c     = len(candles)

    print(f"\n{'='*70}")
    print(f"  Regime v2  —  {now_str}")
    print(f"  {n_c} candle rows  |  {len(rejs)} gate rejections  |  {total} qualifying rows")
    print(f"{'='*70}")

    # ── N+1 accuracy ──────────────────────────────────────────────────────────
    print(f"\n── N+1 Accuracy (regime[N] predicts direction[N+1]) {'─'*19}")
    if n_c < 2:
        print(f"  (need ≥2 candle rows)")
    else:
        # Kalshi benchmark: prefer early_mid[N+1] (T+35s), fall back to open_mid[N+1]
        pairs: list[tuple] = []
        srcs:  list[str]   = []
        for i in range(n_c - 1):
            nxt = candles[i + 1]
            if nxt[5] is not None:
                kv, ks = nxt[5], f"T+{nxt[6]*900:.0f}s"
            else:
                kv, ks = nxt[3], "T=0"
            pairs.append((candles[i][0][:16], candles[i][1], kv, nxt[4]))
            srcs.append(ks)

        t35 = sum(1 for s in srcs if s != "T=0")
        n   = len(pairs)

        rb_all  = sum((p - d)**2 for _, p, k, d in pairs) / n
        kb_all  = sum((k - d)**2 for _, p, k, d in pairs) / n
        acc_all = sum(1 for _, p, k, d in pairs if int(p >= 0.5) == d) / n
        adv_all = (kb_all - rb_all) / kb_all * 100 if kb_all else 0
        beat    = "✓ beating" if rb_all < kb_all else "✗ behind "

        print(f"  All-time  n={n:<3d}  "
              f"regime={rb_all:.3f} ({acc_all:.0%})  "
              f"kalshi={kb_all:.3f}  "
              f"adv={adv_all:+.1f}%  {beat}")

        if n >= 5:
            last = pairs[-5:]
            rb5  = sum((p - d)**2 for _, p, k, d in last) / 5
            kb5  = sum((k - d)**2 for _, p, k, d in last) / 5
            acc5 = sum(1 for _, p, k, d in last if int(p >= 0.5) == d) / 5
            adv5 = (kb5 - rb5) / kb5 * 100 if kb5 else 0
            trend = "improving" if rb5 < rb_all else "declining"
            print(f"  Last-5    n=5    "
                  f"regime={rb5:.3f} ({acc5:.0%})  "
                  f"kalshi={kb5:.3f}  "
                  f"adv={adv5:+.1f}%  {trend}")

        if t35 < n:
            print(f"  Kalshi src: T+35s for {t35}/{n}, T=0 fallback for {n-t35}")

        # Per-candle table — newest at top, pending current candle at very top
        shown  = list(reversed(pairs[-10:]))
        ssrcs  = list(reversed(srcs[-10:]))
        off    = max(0, n - 10)
        nshow  = len(shown)

        print(f"\n  {'Candle [N]':<16}  {'Pred':<6}  N+1  Res   R.Brier  K.Brier  Src")
        print(f"  {'─'*16}  {'─'*6}  {'─'*3}  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*5}")

        # Current open candle — pending row
        cur  = data["cur_regime"]
        prog = data["progress"]
        c_ts = data["candle_open"].strftime("%Y-%m-%dT%H:%M")
        if cur is not None:
            cp, ccoh, cgate = cur
            cpred = f"{'UP  ' if cp >= 0.5 else 'DOWN'} {cp:.2f}"
            coh_s = f"coh={ccoh:.2f}" if ccoh else ""
            print(f"  {c_ts}  {cpred}  ...  ?   ?  {prog*100:.0f}% in  {coh_s}  [OPEN]")
        else:
            print(f"  {c_ts}  (no signal yet)            {prog*100:.0f}% in  [OPEN]")

        for j, ((ts, p, k, d), src) in enumerate(zip(shown, ssrcs)):
            ci   = off + (nshow - 1 - j)
            ok   = int(p >= 0.5) == d
            rb   = (p - d)**2
            kb   = (k - d)**2
            pred = f"{'UP  ' if p >= 0.5 else 'DOWN'} {p:.2f}"
            nxt  = "UP " if d else "DN "
            res  = "✓" if ok else "✗"
            star = "★" if rb < kb else " "
            print(f"  {candles[ci][0][:16]}  {pred}  {nxt}  {res}  {star}  {rb:.3f}    {kb:.3f}   {src}")

        if n < 20:
            notes = []
            if n < 10: notes.append(f"{10-n} more for tier stats")
            notes.append(f"{20-n} more for go-live read")
            print(f"\n  Next: {' · '.join(notes)}")

    # ── Gate rejections ───────────────────────────────────────────────────────
    print(f"\n── Gate Rejections (newest at top) {'─'*35}")

    by_ticker: dict[str, list] = {}
    for r in rejs[:40]:
        by_ticker.setdefault(r[1][-15:], []).append(r)

    res_outcomes = [next((e[7] for e in ent if e[7] is not None), None)
                    for ent in by_ticker.values()]
    wins      = sum(1 for o in res_outcomes if o == 1)
    total_res = sum(1 for o in res_outcomes if o is not None)
    stable    = sum(1 for ent in by_ticker.values()
                    if len(ent) > 1 and
                       max(e[3] for e in ent) - min(e[3] for e in ent) < 0.001)
    multi     = sum(1 for ent in by_ticker.values() if len(ent) > 1)

    if total_res:
        win_pct = wins / total_res
        print(f"  Resolved: {wins}/{total_res} ({win_pct:.0%})  "
              f"| Cache stable: {stable}/{multi} multi-entry markets")

    print(f"\n  {'Time':<5}  {'Market':<15}  {'Regime':<13}  {'Coh':>4}  {'Gates':<9}  Result")
    print(f"  {'─'*5}  {'─'*15}  {'─'*13}  {'─'*4}  {'─'*9}  {'─'*6}")
    for ticker, entries in list(by_ticker.items())[:9]:
        ts      = datetime.fromtimestamp(entries[0][0], tz=timezone.utc).strftime("%H:%M")
        probs   = [e[3] for e in entries]
        cohs    = [e[4] for e in entries if e[4] is not None]
        gates   = ",".join(str(e[2]) for e in entries)
        outcome = next((e[7] for e in entries if e[7] is not None), None)
        avg_coh = sum(cohs) / len(cohs) if cohs else 0
        result  = "WIN    " if outcome == 1 else "LOSS   " if outcome == 0 else "pending"

        stbl = len(probs) > 1 and max(probs) - min(probs) < 0.001
        if len(probs) == 1:
            prob_str = f"{probs[0]:.3f}      "
        elif stbl:
            prob_str = f"{probs[0]:.3f} x{len(probs)} [cache]"
        else:
            prob_str = f"{min(probs):.2f}->{max(probs):.2f} [pre]"

        print(f"  {ts:<5}  {ticker:<15}  {prob_str:<13}  {avg_coh:.2f}  {gates:<9}  {result}")

    # ── Milestones ─────────────────────────────────────────────────────────────
    print(f"\n── Milestones {'─'*56}")

    warm_base = data["warm_trained_rows"] or 682
    warm_left = max(0, warm_base + 50 - total)
    warm_pct  = min(1.0, (total - warm_base) / 50)
    warm_s    = "FIRES NOW" if warm_left == 0 else f"{warm_left} rows (~{warm_left//6}h)"
    print(f"  Warm-start  {_bar(warm_pct)}  {warm_s}")

    cal_n   = min(n_c, 10)
    cal_s   = "ready" if n_c >= 10 else f"{10-n_c} more"
    print(f"  Tier stats  {_bar(cal_n/10)}  {cal_n}/10  {cal_s}")

    seen: set[str] = set()
    uniq = sum(1 for r in rejs
               if r[7] is not None and r[1] not in seen and not seen.add(r[1]))  # type: ignore
    # cleaner:
    seen2: set[str] = set()
    uniq2 = 0
    for r in rejs:
        if r[7] is not None and r[1] not in seen2:
            seen2.add(r[1])
            uniq2 += 1
    print(f"  Phase 3c    {_bar(uniq2/500)}  {uniq2}/500 unique markets (~Day 18-20)")
    print()


if __name__ == "__main__":
    main()
