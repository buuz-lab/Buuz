#!/usr/bin/env python3
"""Kronos live terminal monitor — python3 scripts/live_monitor.py"""

import datetime
import os
import re
import sqlite3
import subprocess
import time

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

DB = "/Users/ezrakornberg/Kronos V2/trades.db"
LOG_DIR = "/Users/ezrakornberg/Kronos V2/logs"
REFRESH = 5

console = Console()


# ── Time / log helpers ────────────────────────────────────────────────────────

def pst_now():
    return datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=8)


def today_pst_epoch() -> int:
    p = pst_now()
    midnight_pst_utc = datetime.datetime(p.year, p.month, p.day, 8, 0, 0,
                                         tzinfo=datetime.timezone.utc)
    return int(midnight_pst_utc.timestamp())


def latest_log():
    try:
        logs = [f for f in os.listdir(LOG_DIR)
                if f.startswith("kronos_") and f.endswith(".log")
                and f != "kronos_stdout.log"]
        if not logs:
            return None
        logs.sort(key=lambda f: os.path.getmtime(os.path.join(LOG_DIR, f)), reverse=True)
        return os.path.join(LOG_DIR, logs[0])
    except Exception:
        return None


def grep_last(path, pattern, n=6):
    if not path or not os.path.exists(path):
        return []
    try:
        out = subprocess.run(["grep", pattern, path],
                             capture_output=True, text=True, timeout=3)
        lines = [l for l in out.stdout.strip().split("\n") if l]
        return lines[-n:]
    except Exception:
        return []


# ── Color helpers ─────────────────────────────────────────────────────────────

def color_prob(val) -> Text:
    try:
        n = float(val)
    except (TypeError, ValueError):
        return Text("—", style="dim")
    s = f"{n:.2f}"
    if n >= 0.70: return Text(s, style="bold green")
    if n >= 0.55: return Text(s, style="green")
    if n <= 0.30: return Text(s, style="bold red")
    if n <= 0.45: return Text(s, style="red")
    return Text(s, style="yellow")


def color_result(outcome) -> Text:
    if outcome == 1:  return Text("WIN",  style="bold green")
    if outcome == 0:  return Text("LOSS", style="bold red")
    return Text("...",  style="yellow dim")


def color_pnl(val) -> Text:
    try:
        n = float(val)
        return Text(f"+${n:.2f}" if n >= 0 else f"${n:.2f}",
                    style="bold green" if n >= 0 else "bold red")
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
        return Text(s, style="bold magenta") if (n <= 35 or n >= 65) else Text(s, style="white")
    except Exception:
        return Text("—", style="dim")


def color_dir(direction) -> Text:
    if direction == 1: return Text("YES→UP",  style="bold green")
    return Text("NO→DOWN", style="bold red")


def color_regime(r) -> Text:
    styles = {"trending_up": "green", "trending_down": "red",
               "high_uncertainty": "yellow", "ranging": "dim"}
    return Text(str(r or "?"), style=styles.get(str(r), "dim"))


def color_gate(gate) -> Text:
    colors = {2: "dim", 3: "yellow", 4: "yellow", 5: "yellow",
               7: "cyan", 8: "magenta", 9: "blue", 10: "red"}
    return Text(f"G{gate}", style=colors.get(gate, "white"))


# ── DB helpers ────────────────────────────────────────────────────────────────

def open_db():
    return sqlite3.connect(f"file:{DB}?mode=ro", uri=True, check_same_thread=False)


def today_filter_trades():
    return ("strftime('%Y-%m-%d', substr(timestamp,1,19), '-8 hours')"
            " = strftime('%Y-%m-%d', 'now', '-8 hours')")


# ── Panel builders ────────────────────────────────────────────────────────────

def make_header() -> Text:
    t = pst_now().strftime("%H:%M:%S  %a %Y-%m-%d  PST")
    out = Text()
    out.append("  KRONOS MONITOR", style="bold cyan")
    out.append(f"  —  {t}  —  refresh {REFRESH}s  ", style="dim")
    return out


def make_bg_panel(log) -> Panel:
    lines = grep_last(log, "KronosBG:", 4)
    t = Table(box=None, show_header=True, header_style="dim",
              expand=True, padding=(0, 2))
    t.add_column("Candle (UTC)",   style="dim",  min_width=20, no_wrap=True)
    t.add_column("k5",             min_width=6,  no_wrap=True)
    t.add_column("k15",            min_width=6,  no_wrap=True)
    t.add_column("Strike",         style="dim",  min_width=10, no_wrap=True)
    for line in lines:
        k5     = re.search(r"prob=([0-9.]+)", line)
        k15    = re.search(r"prob_15min=([0-9.]+)", line)
        candle = re.search(r"candle=(\S+)", line)
        strike = re.search(r"strike=([0-9.]+)", line)
        cv  = candle.group(1) if candle else "?"
        k5v = k5.group(1) if k5 else "?"
        k15v = k15.group(1) if k15 else "?"
        sv  = f"${float(strike.group(1)):,.0f}" if strike else "?"
        bold_k15 = color_prob(k15v)
        bold_k15.stylize("bold")
        t.add_row(cv, color_prob(k5v), bold_k15, sv)
    return Panel(t, title="[bold cyan]BG LOOP[/]  [dim]last 4 candles[/]",
                 border_style="bright_black")


def make_trades_panel(db) -> Panel:
    rows = db.execute(f"""
        SELECT strftime('%H:%M', substr(timestamp,1,19), '-8 hours') as pst,
               ticker, direction, fill_price_cents,
               printf('%.2f', kelly_dollars), kelly_contracts, outcome,
               CASE WHEN outcome=1 THEN ROUND(kelly_contracts*(100-fill_price_cents)/100.0,2)
                    WHEN outcome=0 THEN ROUND(-kelly_contracts*fill_price_cents/100.0,2)
                    ELSE NULL END as pnl
        FROM trades
        WHERE {today_filter_trades()}
        ORDER BY timestamp DESC LIMIT 10
    """).fetchall()

    t = Table(box=box.SIMPLE, show_header=True, header_style="dim",
              expand=True, padding=(0, 1))
    t.add_column("PST",      min_width=5,  no_wrap=True)
    t.add_column("Market",   min_width=8,  no_wrap=True)
    t.add_column("Dir",      min_width=8,  no_wrap=True)
    t.add_column("Fill",     min_width=4,  no_wrap=True)
    t.add_column("Kelly $",  min_width=7,  no_wrap=True)
    t.add_column("Size",     min_width=4,  no_wrap=True)
    t.add_column("P&L",      min_width=7,  no_wrap=True)
    t.add_column("Result",   min_width=5,  no_wrap=True)

    for row in rows:
        pst_t, ticker, direction, fill, kelly, contracts, outcome, pnl = row
        mkt = re.sub(r"KXBTC15M-\d{2}MAY\d{2}", "", ticker)
        t.add_row(
            pst_t or "—",
            mkt or "—",
            color_dir(direction),
            color_fill(fill),
            Text(f"${kelly}", style="yellow"),
            Text(f"{contracts}x", style="dim"),
            color_pnl(pnl) if pnl is not None else Text("—", style="dim"),
            color_result(outcome),
        )
    return Panel(t, title="[bold cyan]TRADES TODAY[/]  [dim]last 10[/]",
                 border_style="bright_black")


def make_rejections_panel(db) -> Panel:
    epoch = today_pst_epoch()
    rows = db.execute("""
        SELECT strftime('%H:%M', datetime(timestamp,'unixepoch'), '-8 hours') as pst,
               failed_gate,
               ROUND(k15_calibrated_prob, 2),
               would_be_fill_cents,
               deepseek_regime,
               outcome
        FROM gate_rejections
        WHERE timestamp >= ?
        ORDER BY timestamp DESC LIMIT 12
    """, (epoch,)).fetchall()

    t = Table(box=box.SIMPLE, show_header=True, header_style="dim",
              expand=True, padding=(0, 1))
    t.add_column("PST",      min_width=5,  no_wrap=True)
    t.add_column("Gate",     min_width=4,  no_wrap=True)
    t.add_column("k15cal",   min_width=6,  no_wrap=True)
    t.add_column("Fill",     min_width=4,  no_wrap=True)
    t.add_column("Regime",   min_width=16, no_wrap=True)
    t.add_column("Result",   min_width=5,  no_wrap=True)

    for row in rows:
        pst_t, gate, k15cal, fill, regime, outcome = row
        t.add_row(
            pst_t or "—",
            color_gate(gate),
            color_prob(k15cal),
            color_fill(fill) if fill else Text("—", style="dim"),
            color_regime(regime),
            color_result(outcome),
        )
    return Panel(t, title="[bold cyan]GATE REJECTIONS TODAY[/]  [dim]last 12[/]",
                 border_style="bright_black")


def make_pnl_panel(db) -> Panel:
    def query(where=""):
        return db.execute(f"""
            SELECT COUNT(*),
                   SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN outcome=0 THEN 1 ELSE 0 END),
                   ROUND(100.0*SUM(CASE WHEN outcome=1 THEN 1 ELSE 0 END)/
                     NULLIF(SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END),0),1),
                   ROUND(SUM(
                     CASE WHEN outcome=1 THEN kelly_contracts*(100-fill_price_cents)/100.0
                          WHEN outcome=0 THEN -kelly_contracts*fill_price_cents/100.0
                          ELSE 0 END), 2)
            FROM trades WHERE outcome IS NOT NULL {where}
        """).fetchone()

    today   = query(f"AND {today_filter_trades()}")
    alltime = query()

    t = Table(box=None, show_header=False, expand=True, padding=(0, 3))
    t.add_column("Label",   min_width=14, style="bold")
    t.add_column("Trades",  min_width=7)
    t.add_column("Record",  min_width=10)
    t.add_column("WR",      min_width=7)
    t.add_column("Net P&L", min_width=10)

    def add_row(label, data, label_style):
        n, w, l, wr, pnl = data
        n = n or 0; w = w or 0; l = l or 0
        record = Text(f"{w}W / {l}L")
        t.add_row(
            Text(label, style=label_style),
            Text(str(n), style="white"),
            record,
            color_wr(wr),
            color_pnl(pnl),
        )

    add_row("Today (PST)", today,   "bold white")
    add_row("All-time",    alltime, "dim")

    return Panel(t, title="[bold cyan]P&L[/]", border_style="bright_black")


def make_regime_panel(log) -> Panel:
    lines = grep_last(log, "regime:features", 1)
    line = lines[-1] if lines else ""
    ds_lines = grep_last(log, "DeepSeek context", 1)
    ds_line = ds_lines[-1] if ds_lines else ""

    def extract(key):
        m = re.search(rf"'{key}': ([-0-9.]+)", line)
        return float(m.group(1)) if m else None

    def signed_text(v, label):
        if v is None: return Text(f"{label}: —", style="dim")
        style = "green" if v >= 0.3 else "red" if v <= -0.3 else "yellow"
        return Text(f"{label}: {v:+.3f}", style=style)

    cvd  = extract("cvd_normalized")
    lp   = extract("large_print_direction")
    fund = extract("funding_rate")
    fg   = re.search(r"'fear_greed_label': '([^']+)'", line)
    ds_r = re.search(r"regime=(\S+)", ds_line)
    ds_c = re.search(r"confidence=([\d.]+)", ds_line)

    t = Table(box=None, show_header=False, expand=True, padding=(0, 3))
    t.add_column(min_width=18)
    t.add_column(min_width=18)
    t.add_column(min_width=20)
    t.add_column(min_width=20)
    t.add_column(min_width=22)

    ds_text = Text("DeepSeek: ", style="dim")
    if ds_r:
        ds_text.append_text(color_regime(ds_r.group(1)))
        if ds_c:
            ds_text.append(f"  conf:{ds_c.group(1)}", style="dim")

    t.add_row(
        signed_text(cvd, "CVD"),
        signed_text(lp,  "LargePrint"),
        Text(f"Funding: {fund:.6f}" if fund else "Funding: —", style="dim"),
        Text(f"Fear/Greed: {fg.group(1) if fg else '—'}", style="dim"),
        ds_text,
    )
    return Panel(t, title="[bold cyan]REGIME[/]", border_style="bright_black")


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    log = latest_log()
    try:
        db = open_db()
    except Exception as e:
        return Panel(Text(f"DB error: {e}", style="red"))

    out = Group(
        make_header(),
        make_bg_panel(log),
        make_trades_panel(db),
        make_rejections_panel(db),
        make_pnl_panel(db),
        make_regime_panel(log),
        Text(f"\n  [dim]Ctrl+C to stop[/dim]"),
    )
    db.close()
    return out


def main():
    with Live(render(), refresh_per_second=1, screen=True, console=console) as live:
        while True:
            time.sleep(REFRESH)
            live.update(render())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/]")
