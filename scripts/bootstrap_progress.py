"""Prints bootstrap progress toward paper-trading thresholds."""
import sqlite3
from pathlib import Path

db_path = Path("trades.db")
if not db_path.exists():
    print("trades.db not found — system has not run yet.")
    raise SystemExit(0)

conn = sqlite3.connect(str(db_path))
total    = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
resolved = conn.execute("SELECT COUNT(*) FROM trades WHERE outcome IS NOT NULL").fetchone()[0]
wins     = conn.execute("SELECT SUM(outcome) FROM trades WHERE outcome IS NOT NULL").fetchone()[0] or 0
open_pos = total - resolved

print(f"Total trades logged    : {total}")
print(f"Open (not yet resolved): {open_pos}")
print(f"Resolved               : {resolved}  (need 500 for calibrator)")
if resolved:
    print(f"  Wins                 : {wins}")
    print(f"  Losses               : {resolved - wins}")
    print(f"  Win rate             : {wins / resolved * 100:.1f}%")
else:
    print("  Win rate             : —")
print(f"Edge tracker window    : {min(resolved, 50)} / 50  (need 30 for gate 4)")
print()
calibrator_pct = min(resolved / 500 * 100, 100)
edge_pct       = min(resolved / 30 * 100, 100)
print(f"Calibrator threshold   : [{('#' * int(calibrator_pct / 5)).ljust(20)}] {calibrator_pct:.0f}%")
print(f"Edge gate threshold    : [{('#' * int(edge_pct / 5)).ljust(20)}] {edge_pct:.0f}%")
if resolved >= 500:
    print("\nREADY TO GO LIVE — set PAPER_TRADING=false in .env and restart.")
