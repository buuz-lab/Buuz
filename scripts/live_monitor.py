#!/usr/bin/env python3
"""Kronos live terminal monitor — python3 scripts/live_monitor.py"""

import datetime
import os
import re
import sqlite3
import subprocess
import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.text import Text
from rich.columns import Columns

DB = "/Users/ezrakornberg/Kronos V2/trades.db"
LOG_DIR = "/Users/ezrakornberg/Kronos V2/logs"
REFRESH = 5

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────────

def pst_now():
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=8)


def today_pst_epoch() -> int:
    pst = pst_now()
    midnight = datetime.datetime(pst.year, pst.month, pst.day, 8, 0, 0,
                                 tzinfo=datetime.timezone.utc)
    return int(midnight.timestamp())


def latest_log():
    try:
        logs = sorted(
            [f for f in os.listdir(LOG_DIR) if f.startswith("kronos_") and f.endswith(".log")],
            reverse=True,
        )
        return os.path.join(LOG_DIR, logs[0]) if logs else None
    except Exception:
        return None


def grep_last(log_path, pattern, n=5):
    if not log_path or not os.path.exists(log_path):
        return []
    try:
        out = subprocess.run(["grep", pattern, log_path], capture_output=True, text=True, timeout=3)
        lines = [l for l in out.stdout.strip().split("\n") if l]
        return lines[-n:]
    except Exception:
        return []


def color_prob(val) -> Text:
    try:
        n = float(val)
    except (TypeError, ValueError):
        return Text(str(val), style="dim")
    s = f"{n:.2f}"
    if n >= 0.70:   return Text(s, style="bold green")
    if n >= 0.55:   return Text(s, style="green")
    if n <= 0.30:   return Text(s, style="bold red")
    if n <= 0.45:   return Text(s, style="red")
    return Text(s, style="yellow")


def color_result(outcome) -> Text:
    if outcome == 1:   return Text("WIN",  style="bold green")
    if outcome == 0:   return Text("LOSS", style="bold red")
    return Text("...", style="yellow")


def color_pnl(val) -> Text:
    try:
        n = float(val)
        if n >= 0: return Text(f"+${n:.2f}", style="bold green")
        return Text(f"${n:.2f}", style="bold red")
    except Exception:
        return Text("—", style="dim")


def color_wr(val) -> Text:
    try:
        n = float(val)
        s = f"{n:.1f}%"
        if n >= 55: return Text(s, style="bold green")
        if n >= 48: return Text(s, style="yellow")
        return Text(s, style="red")
    except Exception:
        return Text("—", style="dim")


def color_fill(fill) -> Text:
    try:
        n = int(fill)
        s = f"{n}¢"
        if n <= 35 or n >= 65: return Text(s, style="bold magenta")
        return Text(s, style="white")
    except Exception:
        return Text("—", style="dim")


def color_dir(direction) -> Text:
    if direction == 1: return Text("YES→UP",  style="green")
    return Text("NO→DOWN", style="red")


def color_regime(r) -> Text:
    styles = {
        "trending_up":       "green",
        "trending_down":     "red",
        "high_uncertainty":  "yellow",
        "ranging":           "dim",
    }
    return Text(str(r), style=styles.get(str(r), "dim"))


def color_gate(gate) -> Text:
    colors = {2: "dim", 3: "yellow", 4: "yellow", 5: "yellow",
              7: "cyan", 8: "magenta", 9: "blue", 10: "red"}
    return Text(f"G{gate}", style=colors.get(gate, "white"))


# ── DB queries ────────────────────────────────────────────────────────────────

def open_db():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True,
                           check_same_thread=False)


def get_trades(db, limit=8):
    return db.execute("""
        SELECT strftime('%H:%M', substr(timestamp,1,19), '-8 hours'),
               ticker, direction, fill_price_cents,
               printf('%.2f', kelly_dollars), kelly_contracts, outcome
        FROM trades
        WHERE strftime('%Y-%m-%d', substr(timestamp,1,19), '-8 hours')
            = strftime('%Y-%m-%d', 'now', '-8 hours')
        ORDER BY timestamp DESC LIMIT ?
    """, (limit,)).fetchall()


def get_rejections(db, limit=10):
    epoch = today_pst_epoch()
    return db.execute("""
        SELECT strftime('%H:%M', datetime(timestamp,'unixepoch'), '-8 hours'),
               failed_gate, ROUND(k15_calibrated_prob,2),
               would_be_fill_cents, deepseek_regime, outcome
        FROM gate_rejections
        WHERE timestamp >= ?
        ORDER BY timestamp DESC LIMIT ?
    """, (epoch, limit)).fetchall()


def get_pnl(db):
    today = db.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END),
               ROUND(100.0*SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END)/
                 NULLIF(SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END),0),1),
               ROUND(SUM(CASE WHEN outcome=1 THEN kelly_contracts*(100-fill_price_cents)/100.0
                              WHEN outcome=0 THEN -kelly_contracts*fill_price_cents/100.0
                              ELSE 0 END),2)
        FROM trades WHERE outcome IS NOT NULL
          AND strftime('%Y-%m-%d', substr(timestamp,1,19), '-8 hours')
            = strftime('%Y-%m-%d', 'now', '-8 hours')
    """).fetchone()
    alltime = db.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END),
               ROUND(100.0*SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END)/
                 NULLIF(SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END),0),1),
               ROUND(SUM(CASE WHEN outcome=1 THEN kelly_contracts*(100-fill_price_cents)/100.0
                              WHEN outcome=0 THEN -kelly_contracts*fill_price_cents/100.0
                              ELSE 0 END),2)
        FROM trades WHERE outcome IS NOT NULL
    """).fetchone()
    return today, alltime


# ── Panels ────────────────────────────────────────────────────────────────────

def make_header() -> Text:
    t = pst_now().strftime("%H:%M:%S PST  %Y-%m-%d")
    return Text(f"  KRONOS MONITOR  ─  {t}  ─  refresh {REFRESH}s  ", style="bold cyan on black")


def make_bg_panel(log) -> Panel:
    lines = grep_last(log, "KronosBG:", 4)
    t = Table(box=None, show_header=False, padding=(0, 1))
    t.add_column("candle", style="dim",  width=18)
    t.add_column("k5",     width=10)
    t.add_column("k15",    width=10)
    t.add_column("strike", style="dim",  width=14)
    for line in lines:
        k5     = re.search(r"prob=([0-9.]+)",        line)
        k15    = re.search(r"prob_15min=([0-9.]+)",  line)
        candle = re.search(r"candle=(\S+)",           line)
        strike = re.search(r"strike=([0-9.]+)",       line)
        cv = (candle.group(1) if candle else "?")[-16:]
        k5v  = k5.group(1) if k5 else "?"
        k15v = k15.group(1) if k15 else "?"
        sv   = f"${float(strike.group(1)):,.0f}" if strike else "?"
        row_k15 = color_prob(k15v)
        row_k15.stylize("bold")
        t.add_row(cv, color_prob(k5v), row_k15, sv)
    return Panel(t, title="[bold cyan]BG LOOP[/]", border_style="bright_black")


def make_trades_panel(db) -> Panel:
    rows = get_trades(db, 8)
    t = Table(box=box.SIMPLE, show_header=True, header_style="dim",
              padding=(0, 1))
    t.add_column("PST",      width=5)
    t.add_column("market",   width=10)
    t.add_column("dir",      width=8)
    t.add_column("fill",     width=5)
    t.add_column("kelly",    width=6)
    t.add_column("size",     width=4)
    t.add_column("result",   width=6)
    for row in rows:
        pst_t, ticker, direction, fill, kelly, contracts, outcome = row
        mkt = re.sub(r"KXBTC15M-\d+MAY\d+", "", ticker)
        t.add_row(
            pst_t, mkt, color_dir(direction),
            color_fill(fill),
            Text(f"${kelly}", style="yellow"),
            Text(f"{contracts}x", style="dim"),
            color_result(outcome),
        )
    return Panel(t, title="[bold cyan]TRADES TODAY[/]", border_style="bright_black")


def make_rejections_panel(db) -> Panel:
    rows = get_rejections(db, 10)
    t = Table(box=box.SIMPLE, show_header=True, header_style="dim",
              padding=(0, 1))
    t.add_column("PST",     width=5)
    t.add_column("gate",    width=4)
    t.add_column("k15cal",  width=6)
    t.add_column("fill",    width=5)
    t.add_column("regime",             width=18)
    t.add_column("result",  width=6)
    for row in rows:
        pst_t, gate, k15cal, fill, regime, outcome = row
        t.add_row(
            pst_t,
            color_gate(gate),
            color_prob(k15cal),
            color_fill(fill) if fill else Text("—", style="dim"),
            color_regime(regime),
            color_result(outcome),
        )
    return Panel(t, title="[bold cyan]GATE REJECTIONS TODAY[/]", border_style="bright_black")


def make_pnl_panel(db) -> Panel:
    today, alltime = get_pnl(db)
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column("label",   width=14)
    t.add_column("trades",  width=8)
    t.add_column("record",  width=12)
    t.add_column("wr",      width=8)
    t.add_column("net",     width=12)

    def row(label, data, label_style="bold"):
        n, w, l, wr, pnl = data
        n = n or 0; w = w or 0; l = l or 0
        record = Text(f"{w}W / {l}L", style="default")
        t.add_row(
            Text(label, style=label_style),
            Text(str(n), style="white"),
            record,
            color_wr(wr),
            color_pnl(pnl),
        )

    row("Today (PST)", today, "bold")
    row("All-time",    alltime, "dim")
    return Panel(t, title="[bold cyan]P&L[/]", border_style="bright_black")


def make_regime_panel(log) -> Panel:
    lines = grep_last(log, "regime:features", 1)
    line = lines[-1] if lines else ""
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column(width=16)
    t.add_column(width=16)
    t.add_column(width=16)
    t.add_column(width=24)

    def signed(key, label):
        m = re.search(rf"'{key}': ([-0-9.]+)", line)
        if not m: return Text(f"{label}:?", style="dim")
        v = float(m.group(1))
        style = "green" if v >= 0.3 else "red" if v <= -0.3 else "yellow"
        return Text(f"{label}:{v:.3f}", style=style)

    cvd  = signed("cvd_normalized", "CVD")
    lp   = signed("large_print_direction", "LP")
    fund = re.search(r"'funding_rate': ([-0-9.]+)", line)
    fg   = re.search(r"'fear_greed_label': '([^']+)'", line)
    ds   = grep_last(log, "DeepSeek context", 1)
    ds_regime = re.search(r"regime=(\S+)", ds[-1]) if ds else None
    ds_text = color_regime(ds_regime.group(1)) if ds_regime else Text("?", style="dim")

    t.add_row(
        cvd, lp,
        Text(f"fund:{fund.group(1)[:9] if fund else '?'}", style="dim"),
        Text(f"fear/greed: {fg.group(1) if fg else '?'}", style="dim"),
    )
    regime_row = Table(box=None, show_header=False, padding=(0,2))
    regime_row.add_column(width=20)
    regime_row.add_row(Text("DeepSeek: ", style="dim") + ds_text)
    return Panel(
        Columns([t, regime_row]),
        title="[bold cyan]REGIME[/]",
        border_style="bright_black",
    )


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    log = latest_log()
    try:
        db = open_db()
    except Exception as e:
        return Panel(Text(f"DB error: {e}", style="red"), title="ERROR")

    layout = Layout()
    layout.split_column(
        Layout(name="header",     size=1),
        Layout(name="bg",         size=8),
        Layout(name="middle",     size=18),
        Layout(name="pnl",        size=6),
        Layout(name="regime",     size=5),
    )
    layout["middle"].split_row(
        Layout(name="trades",     ratio=1),
        Layout(name="rejections", ratio=1),
    )

    layout["header"].update(make_header())
    layout["bg"].update(make_bg_panel(log))
    layout["trades"].update(make_trades_panel(db))
    layout["rejections"].update(make_rejections_panel(db))
    layout["pnl"].update(make_pnl_panel(db))
    layout["regime"].update(make_regime_panel(log))

    db.close()
    return layout


def main():
    with Live(render(), refresh_per_second=1, screen=True) as live:
        while True:
            time.sleep(REFRESH)
            live.update(render())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/]")
