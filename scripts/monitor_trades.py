"""
Live monitor for Kronos trades and resolutions.
Polls trades.db every 15 seconds and prints new entries/resolutions.
Runs regime health check every 15 minutes.

Usage:
    python3 scripts/monitor_trades.py
"""
import io
import sqlite3
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.regime_health_check as _rhc

DB = "/Users/ezrakornberg/Kronos V2/trades.db"
HEALTH_CHECK_INTERVAL = 900  # 15 minutes

TRAINING_TARGET = 500

ALL_21_FEATURES = """
  features_stale=0
  AND funding_rate IS NOT NULL AND funding_rate_trend IS NOT NULL
  AND oi_delta_pct IS NOT NULL AND cvd_normalized IS NOT NULL
  AND basis_spread_pct IS NOT NULL AND brti_volatility_1h IS NOT NULL
  AND cvd_velocity IS NOT NULL AND cvd_acceleration IS NOT NULL
  AND brti_momentum_5min IS NOT NULL AND brti_momentum_15min IS NOT NULL
  AND candle_progress IS NOT NULL AND hour_sin IS NOT NULL AND hour_cos IS NOT NULL
  AND kalshi_implied_prob IS NOT NULL AND funding_window_proximity IS NOT NULL
  AND trend_slope_1h IS NOT NULL AND trend_r2_1h IS NOT NULL
  AND hourly_sr_proximity IS NOT NULL AND range_breakout_flag IS NOT NULL
  AND tape_speed_tpm IS NOT NULL AND large_print_direction IS NOT NULL
"""


def fmt_ts(ts):
    return ts[:10] + " " + ts[11:19] + " UTC"


def get_stats(conn):
    row = conn.execute(f"""
        SELECT
            COUNT(*),
            COUNT(*) FILTER (WHERE outcome IS NOT NULL),
            SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END) FILTER (WHERE outcome IS NOT NULL),
            COALESCE(SUM(pnl_dollars) FILTER (WHERE outcome IS NOT NULL), 0),
            COUNT(*) FILTER (WHERE {ALL_21_FEATURES} AND outcome IS NOT NULL)
        FROM trades
    """).fetchone()
    total, resolved, wins, net_pnl, tr = row
    wins = int(wins or 0)
    losses = resolved - wins
    open_pos = total - resolved
    pct = int(wins * 100 / resolved) if resolved > 0 else 0
    tr_pct = min(int(tr * 100 / TRAINING_TARGET), 100)
    bar = "#" * (tr_pct // 5) + "-" * (20 - tr_pct // 5)
    return total, resolved, wins, losses, net_pnl, tr, open_pos, pct, tr_pct, bar


def get_streak(conn):
    rows = conn.execute("""
        SELECT outcome FROM trades
        WHERE outcome IS NOT NULL
        ORDER BY timestamp DESC LIMIT 50
    """).fetchall()
    if not rows:
        return 0, None
    first_win = rows[0][0] == 1
    count = 0
    for (outcome,) in rows:
        if (outcome == 1) == first_win:
            count += 1
        else:
            break
    return count, "W" if first_win else "L"


def get_open_trades(conn):
    return conn.execute("""
        SELECT timestamp, ticker, direction, kelly_contracts, fill_price_cents, kelly_dollars
        FROM trades WHERE outcome IS NULL ORDER BY timestamp ASC
    """).fetchall()


def stat_line(wins, losses, pct, open_pos, net_pnl, tr, bar, tr_pct, streak_n, streak_type):
    streak = f"  Streak: {streak_n}{streak_type}" if streak_type else ""
    return (
        f"    RECORD: {wins}W / {losses}L ({pct}%){streak}"
        f"  Open: {open_pos}  Net P&L: ${net_pnl:.2f}"
        f"  |  TRAINING: {tr}/{TRAINING_TARGET} [{bar}] {tr_pct}%"
    )


def print_open_trades(conn):
    rows = get_open_trades(conn)
    if not rows:
        return
    for ts, ticker, direction, contracts, price, kelly in rows:
        arrow = "YES→UP" if direction == 1 else "NO→DOWN"
        print(f"    OPEN POSITION: {ticker} {arrow} {contracts}x{price}¢  (${kelly:.2f} kelly)  placed {fmt_ts(ts)}", flush=True)


def print_last5(conn):
    rows = conn.execute("""
        SELECT timestamp, ticker, direction, kelly_contracts, fill_price_cents, kelly_dollars, outcome, pnl_dollars
        FROM trades ORDER BY timestamp DESC LIMIT 5
    """).fetchall()
    print("  Last 5 trades:", flush=True)
    print(f"  {'Time (UTC)':<20} {'Market':<28} {'Dir':<9} {'Size':<8} {'Kelly':>7}  Result", flush=True)
    print("  " + "-" * 85, flush=True)
    for ts, ticker, direction, contracts, price, kelly, outcome, pnl in rows:
        dir_str = "YES→UP" if direction == 1 else "NO→DOWN"
        if outcome is None:
            result = "OPEN"
        elif outcome == 1:
            result = f"WIN  ${pnl:.2f}"
        else:
            result = f"LOSS ${pnl:.2f}"
        print(f"  {ts[:19]:<20} {ticker:<28} {dir_str:<9} {contracts}x{price}¢{'':<2} ${kelly:>6.2f}  {result}", flush=True)


def run_health_check():
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*60}", flush=True)
    print(f"  REGIME HEALTH CHECK  [{now_str}]", flush=True)
    print(f"{'='*60}", flush=True)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            _rhc.main()
        for line in buf.getvalue().splitlines():
            print(f"  {line}", flush=True)
    except Exception as exc:
        print(f"  health check error: {exc}", flush=True)
    print(f"{'='*60}\n", flush=True)


prev_total = -1
prev_resolved = -1
last_health_check = 0  # run immediately on startup

while True:
    try:
        conn = sqlite3.connect(DB)
        total, resolved, wins, losses, net_pnl, tr, open_pos, pct, tr_pct, bar = get_stats(conn)
        streak_n, streak_type = get_streak(conn)

        now = time.time()
        if now - last_health_check >= HEALTH_CHECK_INTERVAL:
            run_health_check()
            last_health_check = now

        if total != prev_total and prev_total >= 0:
            new = conn.execute(f"""
                SELECT timestamp, ticker, direction, kelly_contracts, fill_price_cents, kelly_dollars
                FROM trades ORDER BY timestamp DESC LIMIT {total - prev_total}
            """).fetchall()
            for ts, ticker, direction, contracts, cents, kelly in reversed(new):
                arrow = "YES→UP" if direction == 1 else "NO→DOWN"
                print(f"*** TRADE ENTERED [{fmt_ts(ts)}] {ticker} {arrow} {contracts}x{cents}¢  (${kelly:.2f} kelly)", flush=True)
            print(stat_line(wins, losses, pct, open_pos, net_pnl, tr, bar, tr_pct, streak_n, streak_type), flush=True)
            print_open_trades(conn)
            print_last5(conn)

        if resolved != prev_resolved and prev_resolved >= 0:
            new = conn.execute(f"""
                SELECT timestamp, ticker, direction, outcome, pnl_dollars
                FROM trades WHERE outcome IS NOT NULL ORDER BY timestamp DESC LIMIT {resolved - prev_resolved}
            """).fetchall()
            for ts, ticker, direction, outcome, pnl in reversed(new):
                arrow = "YES→UP" if direction == 1 else "NO→DOWN"
                result = "WIN" if outcome == 1 else "LOSS"
                print(f"*** RESOLVED [placed {fmt_ts(ts)}] {ticker} {arrow} => {result}  pnl=${pnl:.2f}", flush=True)
            print(stat_line(wins, losses, pct, open_pos, net_pnl, tr, bar, tr_pct, streak_n, streak_type), flush=True)
            print_open_trades(conn)

        if prev_total == -1:
            print(f"=== MONITOR STARTED ===", flush=True)
            print(stat_line(wins, losses, pct, open_pos, net_pnl, tr, bar, tr_pct, streak_n, streak_type), flush=True)
            print_open_trades(conn)
            print_last5(conn)

        conn.close()
        prev_total = total
        prev_resolved = resolved

    except Exception as e:
        print(f"DB error: {e}", flush=True)

    time.sleep(15)
