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

# ── ANSI colours ─────────────────────────────────────────────────────────────
G  = "\033[92m"   # green
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow / gold
DIM= "\033[2m"    # dim grey
B  = "\033[94m"   # blue
RST= "\033[0m"    # reset

def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RST}"

def _bar(frac: float, width: int = 22) -> str:
    filled = int(round(frac * width))
    return _c("█" * filled, G) + _c("░" * (width - filled), DIM)


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
        candles = list(reversed(rows))   # chronological, oldest→newest

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
    print(f"\n── N+1 Candle Accuracy {_c('(regime[N] → direction[N+1])', DIM)}  {'─'*20}")
    if n_c < 2:
        print(f"  {_c('(need ≥2 candle rows)', DIM)}")
    else:
        # Kalshi benchmark: prefer kalshi_early_mid[N+1] (T+35s), fall back to open_mid[N+1]
        def _kb(row: tuple) -> tuple[float, str]:
            return (row[5], f"T+{row[6]*900:.0f}s") if row[5] is not None else (row[3], "T=0")

        pairs: list[tuple] = []
        srcs:  list[str]   = []
        for i in range(n_c - 1):
            kv, ks = _kb(candles[i + 1])
            pairs.append((candles[i][0][:16], candles[i][1], kv, candles[i + 1][4]))
            srcs.append(ks)

        t35 = sum(1 for s in srcs if s != "T=0")
        if t35 < len(srcs):
            print(f"  {_c(f'Kalshi: T+35s for {t35}/{len(srcs)} pairs, T=0 fallback for rest', DIM)}")

        n       = len(pairs)
        rb_all  = sum((p - d) ** 2 for _, p, k, d in pairs) / n
        kb_all  = sum((k - d) ** 2 for _, p, k, d in pairs) / n
        acc_all = sum(1 for _, p, k, d in pairs if int(p >= 0.5) == d) / n
        adv_all = (kb_all - rb_all) / kb_all * 100 if kb_all else 0
        beat_c  = G if rb_all < kb_all else R
        beat_s  = "✓ beating Kalshi" if rb_all < kb_all else "✗ behind Kalshi"

        print(f"  All-time  n={n:<3d}  "
              f"regime={_c(f'{rb_all:.3f}', G if rb_all < 0.25 else Y)}  "
              f"({_c(f'{acc_all:.0%}', G if acc_all > 0.55 else R)})  "
              f"kalshi={kb_all:.3f}  "
              f"adv={_c(f'{adv_all:+.1f}%', beat_c)}  "
              f"{_c(beat_s, beat_c)}")

        if n >= 5:
            last  = pairs[-5:]
            rb5   = sum((p - d) ** 2 for _, p, k, d in last) / 5
            kb5   = sum((k - d) ** 2 for _, p, k, d in last) / 5
            acc5  = sum(1 for _, p, k, d in last if int(p >= 0.5) == d) / 5
            adv5  = (kb5 - rb5) / kb5 * 100 if kb5 else 0
            trend = _c("↑ improving", G) if rb5 < rb_all else _c("↓ declining", R)
            print(f"  Last-5    n=5    "
                  f"regime={rb5:.3f} ({acc5:.0%})  "
                  f"kalshi={kb5:.3f}  "
                  f"adv={adv5:+.1f}%  {trend}")

        # Per-candle table — NEWEST AT TOP
        shown_pairs = list(reversed(pairs[-10:]))
        shown_srcs  = list(reversed(srcs[-10:]))
        shown_base  = max(0, n - 10)   # index into candles[] for the oldest shown pair

        print(f"\n  {'Candle [N]':<16}  Pred   N+1  {'Res':^3}  R.Brier  K.Brier  Src")
        print(f"  {'─'*16}  {'─'*6}  {'─'*3}  {'─'*3}  {'─'*7}  {'─'*7}  {'─'*5}")
        for j, ((ts, p, k, d), src) in enumerate(zip(shown_pairs, shown_srcs)):
            # j=0 is newest; candle index = shown_base + (len(shown_pairs)-1-j)
            ci   = shown_base + (len(shown_pairs) - 1 - j)
            ok   = int(p >= 0.5) == d
            rb   = (p - d) ** 2
            kb   = (k - d) ** 2
            pred = f"{'↑' if p >= 0.5 else '↓'} {p:.2f}"
            nxt  = f"{'↑' if d else '↓'}"
            win  = _c("✓", G) if ok else _c("✗", R)
            star = _c("★", Y) if rb < kb else " "
            rb_s = _c(f"{rb:.3f}", G) if rb < kb else f"{rb:.3f}"
            kb_s = _c(f"{kb:.3f}", G) if kb < rb else f"{kb:.3f}"
            src_s = _c(src, G) if src != "T=0" else _c(src, DIM)
            print(f"  {candles[ci][0][:16]}  {pred}  {nxt}    {win}  {star} {rb_s}   {kb_s}  {src_s}")

        if n < 20:
            notes = []
            if n < 10: notes.append(f"{10-n} more for tier stats")
            if n < 20: notes.append(f"{20-n} more for go-live read")
            print(f"\n  {_c('Next: ' + ' · '.join(notes), DIM)}")

    # ── Gate rejections ───────────────────────────────────────────────────────
    print(f"\n── Gate Rejections {_c('(same-candle, newest at top)', DIM)}  {'─'*27}")

    by_ticker: dict[str, list] = {}
    for r in rejs[:40]:
        by_ticker.setdefault(r[1][-15:], []).append(r)

    res_outcomes = [
        next((e[7] for e in entries if e[7] is not None), None)
        for entries in by_ticker.values()
    ]
    wins      = sum(1 for o in res_outcomes if o == 1)
    total_res = sum(1 for o in res_outcomes if o is not None)
    stable    = sum(
        1 for entries in by_ticker.values()
        if len(entries) > 1 and max(e[3] for e in entries) - min(e[3] for e in entries) < 0.001
    )
    multi = sum(1 for e in by_ticker.values() if len(e) > 1)

    if total_res:
        w_c = G if wins / total_res >= 0.5 else R
        print(f"  Resolved: {_c(f'{wins}/{total_res} wins ({wins/total_res:.0%})', w_c)}  "
              f"| {_c(f'Cache stable: {stable}/{multi}', G if stable == multi else Y)}")

    print(f"\n  {'Time':<5}  {'Market':<15}  {'Regime':<14}  {'Coh':>4}  {'Gates':<9}  Outcome")
    print(f"  {'─'*5}  {'─'*15}  {'─'*14}  {'─'*4}  {'─'*9}  {'─'*7}")
    for ticker, entries in list(by_ticker.items())[:9]:
        ts      = datetime.fromtimestamp(entries[0][0], tz=timezone.utc).strftime("%H:%M")
        probs   = [e[3] for e in entries]
        cohs    = [e[4] for e in entries if e[4] is not None]
        gates   = ",".join(str(e[2]) for e in entries)
        outcome = next((e[7] for e in entries if e[7] is not None), None)
        avg_coh = sum(cohs) / len(cohs) if cohs else 0

        result_s = (_c("WIN ", G)  if outcome == 1 else
                    _c("LOSS", R)  if outcome == 0 else
                    _c("pend", DIM))

        stable_mkt = len(probs) > 1 and max(probs) - min(probs) < 0.001
        if len(probs) == 1:
            prob_str = f"{probs[0]:.3f}    "
        elif stable_mkt:
            prob_str = _c(f"{probs[0]:.3f} ×{len(probs)}✓", G)
        else:
            prob_str = _c(f"{min(probs):.2f}→{max(probs):.2f}", DIM) + " pre"

        coh_c = G if avg_coh > 0.65 else (Y if avg_coh > 0.55 else DIM)
        print(f"  {ts:<5}  {ticker:<15}  {prob_str:<14}  "
              f"{_c(f'{avg_coh:.2f}', coh_c)}  {gates:<9}  {result_s}")

    # ── Milestones ─────────────────────────────────────────────────────────────
    print(f"\n── Milestones  {'─'*55}")

    warm_base = data["warm_trained_rows"] or 682
    warm_next = warm_base + 50
    warm_left = max(0, warm_next - total)
    warm_pct  = min(1.0, (total - warm_base) / 50)
    warm_s    = _c("FIRES NOW", G) if warm_left == 0 else f"{warm_left} rows (~{warm_left//6}h)"
    print(f"  Warm-start  {_bar(warm_pct)} {warm_s}")

    cal_pct = min(1.0, n_c / 10)
    cal_s   = _c("ready", G) if n_c >= 10 else f"{10 - n_c} more"
    print(f"  Tier stats  {_bar(cal_pct)} {min(n_c, 10)}/10 candles  {cal_s}")

    seen: set[str] = set()
    uniq = sum(1 for r in rejs if r[7] is not None and not seen.add(r[1]) and r[1] not in seen)
    # simpler unique resolved count
    seen2: set[str] = set()
    uniq = 0
    for r in rejs:
        if r[7] is not None and r[1] not in seen2:
            seen2.add(r[1])
            uniq += 1
    c3_pct = min(1.0, uniq / 500)
    print(f"  Phase 3c    {_bar(c3_pct)} {uniq}/500 unique markets (~Day 18-20)")
    print()


if __name__ == "__main__":
    main()
