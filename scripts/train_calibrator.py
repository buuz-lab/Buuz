"""
Train the Calibrator from instrumented trades in trades.db.

Usage:
    python3 scripts/train_calibrator.py [--db trades.db] [--out models/calibrator.pkl]
                                        [--window 300] [--min-rows 300] [--dry-run]

Label semantics
---------------
The calibrator maps kronos_raw (P(market UP)) to a calibrated probability.
It must be trained with y_up = int(direction == outcome), not raw outcome:

    direction=0, outcome=0 (NO loss = market UP)   → y_up=1 ✓
    direction=0, outcome=1 (NO win  = market DOWN)  → y_up=0 ✓
    direction=1, outcome=1 (YES win = market UP)    → y_up=1 ✓
    direction=1, outcome=0 (YES loss = market DOWN) → y_up=0 ✓

This matches the calibrator's role: predict P(market UP) from Kronos raw output.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from btc_kalshi_system.models.calibrator import Calibrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="trades.db")
    p.add_argument("--out", default=config.CALIBRATOR_MODEL_PATH)
    p.add_argument("--window", type=int, default=300,
                   help="Number of most-recent training-ready rows to use (default: 300)")
    p.add_argument("--min-rows", type=int, default=300)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.db).exists():
        sys.exit(f"Database not found: {args.db}")

    conn = sqlite3.connect(args.db)
    try:
        rows = conn.execute(
            "SELECT kronos_raw_15min, direction, outcome FROM trades "
            "WHERE outcome IS NOT NULL AND features_stale=0 AND kronos_raw_15min IS NOT NULL "
            "ORDER BY timestamp DESC LIMIT ?",
            (args.window,),
        ).fetchall()
    finally:
        conn.close()

    n = len(rows)
    print(f"Training-ready rows (with kronos_raw_15min) in {args.db}: {n}")
    if n < 200:
        print(f"WARNING: only {n} rows have kronos_raw_15min — data is sparse, calibration may be unreliable")

    if n < args.min_rows:
        sys.exit(
            f"Need ≥{args.min_rows} rows to fit calibrator; have {n}. "
            f"Continue running paper trading and re-run later."
        )

    raw_probs = np.array([r[0] for r in rows], dtype=float)
    directions = np.array([r[1] for r in rows], dtype=float)
    outcomes = np.array([r[2] for r in rows], dtype=float)
    y_up = (directions == outcomes).astype(float)

    # Load existing calibrator for pre-retrain Brier comparison
    pre_brier: float | None = None
    if Path(args.out).exists():
        try:
            existing = Calibrator.load(args.out)
            pre_brier = existing.brier_score(raw_probs, y_up)
            print(f"Existing calibrator: n_samples={existing.n_samples} passthrough={existing._passthrough}")
            print(f"Pre-retrain Brier:  {pre_brier:.4f}")
        except Exception as exc:
            print(f"Could not load existing calibrator: {exc}")
    else:
        print(f"No existing calibrator at {args.out} — fitting fresh")

    cal = Calibrator()
    cal.fit(raw_probs, y_up)
    post_brier = cal.brier_score(raw_probs, y_up)

    print(f"Post-retrain Brier: {post_brier:.4f}")
    print(f"Passthrough: {cal._passthrough}")
    print(f"n_samples:   {cal.n_samples}")

    if pre_brier is not None and post_brier > pre_brier:
        print(f"WARNING: new Brier {post_brier:.4f} > old Brier {pre_brier:.4f} — calibration degraded")

    if args.dry_run:
        print("\n--dry-run set — calibrator NOT saved.")
        return

    os.makedirs("models", exist_ok=True)
    cal.save(args.out)
    print(f"\nSaved calibrator to: {args.out}")


if __name__ == "__main__":
    main()
