"""
Train the Calibrator from instrumented trades + gate_rejections in trades.db.

Usage:
    python3 scripts/train_calibrator.py [--db trades.db] [--out models/calibrator.pkl]
                                        [--window 300] [--min-rows 100] [--dry-run]

Data sources
------------
Training rows are drawn from both tables to eliminate selection bias:
  - trades: placed trades with resolved outcomes
  - gate_rejections: blocked signals with resolved counterfactual outcomes
    (shadow=1 Gate 7 rows excluded via WHERE shadow=0)

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
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.db).exists():
        sys.exit(f"Database not found: {args.db}")

    _UNION_QUERY = """
        SELECT kronos_raw_15min, direction, outcome, timestamp, deepseek_regime FROM (
            SELECT kronos_raw_15min, direction, outcome, timestamp, deepseek_regime FROM trades
            WHERE outcome IS NOT NULL
              AND kronos_raw_15min IS NOT NULL
            UNION ALL
            SELECT kronos_raw_15min, direction, outcome, timestamp, deepseek_regime FROM gate_rejections
            WHERE outcome IS NOT NULL
              AND kronos_raw_15min IS NOT NULL
              AND shadow = 0
        )
        ORDER BY timestamp DESC LIMIT ?
    """
    _COUNT_QUERY = """
        SELECT COUNT(*) FROM (
            SELECT kronos_raw_15min FROM trades
            WHERE outcome IS NOT NULL AND kronos_raw_15min IS NOT NULL
            UNION ALL
            SELECT kronos_raw_15min FROM gate_rejections
            WHERE outcome IS NOT NULL AND kronos_raw_15min IS NOT NULL
              AND shadow = 0
        )
    """

    conn = sqlite3.connect(args.db)
    try:
        total_available = conn.execute(_COUNT_QUERY).fetchone()[0]
        rows = conn.execute(_UNION_QUERY, (args.window,)).fetchall()
    finally:
        conn.close()

    n = len(rows)
    print(f"Training-ready rows (trades + gate_rejections) in {args.db}: {total_available} available, using {n}")
    if total_available < 200:
        print(f"WARNING: only {total_available} combined rows — data is sparse, calibration may be unreliable")

    if total_available < args.min_rows:
        sys.exit(
            f"Need ≥{args.min_rows} rows to fit calibrator; have {total_available}. "
            f"Continue running paper trading and re-run later."
        )

    raw_probs = np.array([r[0] for r in rows], dtype=float)
    directions = np.array([r[1] for r in rows], dtype=float)
    outcomes = np.array([r[2] for r in rows], dtype=float)
    regimes = np.array([r[4] for r in rows], dtype=object)
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
    cal.fit(raw_probs, y_up, regimes=regimes)
    post_brier = cal.brier_score(raw_probs, y_up)

    print(f"Post-retrain Brier: {post_brier:.4f}")
    print(f"Passthrough: {cal._passthrough}")
    print(f"n_samples:   {cal.n_samples}")

    if pre_brier is not None and post_brier > pre_brier:
        print(f"WARNING: new Brier {post_brier:.4f} > old Brier {pre_brier:.4f} — calibration degraded")

    # Compression map: shows how aggressively the calibrator squashes strong k15 signals.
    # Watch k15_raw=0.80 → k15_cal: if it rises above ~0.65, the calibrator is loosening
    # (more data is teaching it to trust strong signals). If it stays < 0.60, compression
    # is still severe and fusion + regime shrinks will produce near-zero edge.
    checkpoints = [0.60, 0.70, 0.80, 0.90, 1.00]
    print("\nCompression map (k15_raw → k15_cal, neutral regime):")
    for raw in checkpoints:
        cal_val = cal.transform(raw, regime="trending_up")
        bar = "█" * int((cal_val - 0.50) * 200)
        flag = "  ← loosening!" if cal_val > 0.65 and raw >= 0.80 else ""
        print(f"  {raw:.2f} → {cal_val:.4f}  {bar}{flag}")

    if args.dry_run:
        print("\n--dry-run set — calibrator NOT saved.")
        return

    os.makedirs("models", exist_ok=True)
    cal.save(args.out)
    print(f"\nSaved calibrator to: {args.out}")

    import datetime, json
    meta = {
        "trained_at_rows": n,
        "trained_at_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "total_rows_at_train": total_available,
    }
    meta_path = Path(args.out).parent / "calibrator_last_trained.json"
    meta_path.write_text(json.dumps(meta, indent=4))
    print(f"Metadata written to: {meta_path}")


if __name__ == "__main__":
    main()
