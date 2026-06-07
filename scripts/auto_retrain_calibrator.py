# Suggested crontab entry (runs every 2 hours):
# 0 */2 * * * cd "/Users/ezrakornberg/Kronos V2" && source .env && python3 scripts/auto_retrain_calibrator.py >> logs/auto_retrain_calibrator.log 2>&1
#
# Phase 3c: calibrator now trains on regime_prob + signal_edge (not k15_raw).
# Only fires once regime v2 is deployed and 200+ rows have regime_prob populated.
#
# Retraining triggers (in priority order):
#   1. Emergency: Brier score on last 50 regime_prob rows > 0.25
#   2. Row-based: +50 new regime_prob rows since last train
#   3. Time-based: 7 days elapsed since last train
#
# Minimum rows: 200 regime_prob rows (rows where regime_prob IS NOT NULL).
# Before regime v2 deploys all regime_prob values are NULL — auto-retrain is a no-op.
"""
Auto-retrain script for the Kronos V2 calibrator.

Evaluates retraining triggers and, when any fires, invokes train_calibrator.py
as a subprocess.  Designed to be run on a cron schedule (see comment above).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from btc_kalshi_system.models.calibrator import Calibrator

# ── Constants ─────────────────────────────────────────────────────────────────

_MARKER_PATH = "models/calibrator_last_trained.json"

_ROW_TRIGGER_DELTA = 50            # retrain when +50 new regime_prob rows since last train
_TIME_TRIGGER_DAYS = 7             # retrain if 7 days elapsed since last train
_MIN_ROWS = 200                    # refuse to retrain below this — requires regime v2 deployed
_WINDOW = 300                      # rolling window passed to train_calibrator.py
_EMERGENCY_BRIER_THRESHOLD = 0.25  # worse than near-coin-flip → emergency retrain


# ── Helper functions ──────────────────────────────────────────────────────────

def get_k15_ready_count(db_path: str) -> int:
    """Return combined COUNT(*) of Phase 3c-ready rows (regime_prob IS NOT NULL)."""
    if not Path(db_path).exists():
        sys.exit(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM ("
            "    SELECT regime_prob FROM trades"
            "    WHERE outcome IS NOT NULL AND regime_prob IS NOT NULL AND signal_edge IS NOT NULL"
            "    UNION ALL"
            "    SELECT regime_prob FROM gate_rejections"
            "    WHERE outcome IS NOT NULL AND regime_prob IS NOT NULL AND signal_edge IS NOT NULL"
            "      AND shadow = 0"
            ")"
        ).fetchone()[0]
    finally:
        conn.close()
    return int(count)


def compute_emergency_brier(db_path: str, model_path: str) -> tuple[float, bool] | None:
    """
    Load the deployed calibrator and evaluate Brier on the last 50 k15-ready rows.

    Returns (brier, is_passthrough) or None if the model file does not exist.
    Skip the emergency check when calibrator is passthrough (no baseline to compare).
    """
    if not Path(model_path).exists():
        return None

    try:
        cal = Calibrator.load(model_path)
    except Exception:
        return None

    if cal._passthrough:
        return None, True

    if not Path(db_path).exists():
        return None

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT regime_prob, direction, outcome FROM trades"
            " WHERE outcome IS NOT NULL"
            "   AND regime_prob IS NOT NULL"
            " ORDER BY timestamp DESC LIMIT 50"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    regime_probs = np.array([r[0] for r in rows], dtype=float)
    directions   = np.array([r[1] for r in rows], dtype=float)
    outcomes     = np.array([r[2] for r in rows], dtype=float)
    y_yes = np.where(directions == 1, outcomes, 1.0 - outcomes)

    brier = cal.brier_score(regime_probs, y_yes)
    return float(brier), False


def load_marker() -> dict | None:
    """Read _MARKER_PATH; return None if file is missing or corrupt."""
    p = Path(_MARKER_PATH)
    if not p.exists():
        return None
    try:
        with p.open() as f:
            data = json.load(f)
        _ = data["trained_at_rows"], data["trained_at_timestamp"]
        return data
    except (json.JSONDecodeError, KeyError):
        print(f"WARNING: marker file {_MARKER_PATH} is corrupt or incomplete — treating as absent.")
        return None


def save_marker(trained_at_rows: int, total_rows: int) -> None:
    """Write _MARKER_PATH with current state."""
    p = Path(_MARKER_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "trained_at_rows": trained_at_rows,
        "trained_at_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rows_at_train": total_rows,
    }
    with p.open("w") as f:
        json.dump(data, f, indent=4)


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default="trades.db",
                   help="Path to trades.db (default: trades.db)")
    p.add_argument("--out", default=config.CALIBRATOR_MODEL_PATH,
                   help=f"Output path for trained calibrator (default: {config.CALIBRATOR_MODEL_PATH})")
    p.add_argument("--force", action="store_true",
                   help="Bypass all trigger checks and retrain unconditionally.")
    p.add_argument("--dry-run", action="store_true",
                   help="Evaluate triggers and print what would happen without retraining.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 1. Get current k15-ready row count
    count = get_k15_ready_count(args.db)

    # 2. Load marker
    marker = load_marker()

    # 3. Evaluate EMERGENCY trigger
    emergency_result = compute_emergency_brier(args.db, args.out)
    if emergency_result is None:
        emergency_trigger = False
        emergency_detail = "no model deployed"
        emergency_passthrough = False
    else:
        brier_val, is_passthrough = emergency_result
        emergency_passthrough = is_passthrough
        if is_passthrough:
            emergency_trigger = False
            emergency_detail = "calibrator is passthrough — no baseline"
        else:
            emergency_trigger = brier_val > _EMERGENCY_BRIER_THRESHOLD
            emergency_detail = f"Brier {brier_val:.4f} {'>' if emergency_trigger else '<='} {_EMERGENCY_BRIER_THRESHOLD}"

    # 4. Evaluate ROW trigger
    last_trained_rows = marker["trained_at_rows"] if marker else 0
    row_trigger = count >= last_trained_rows + _ROW_TRIGGER_DELTA

    # 5. Evaluate TIME trigger
    if marker:
        last_ts = datetime.fromisoformat(marker["trained_at_timestamp"])
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        elapsed_days = (datetime.now(timezone.utc) - last_ts).total_seconds() / 86400
        time_trigger = elapsed_days >= _TIME_TRIGGER_DAYS
    else:
        elapsed_days = None
        time_trigger = True  # Never trained = time trigger fires

    # 6. Print status header
    last_ts_str = marker["trained_at_timestamp"] if marker else "never"
    elapsed_str = f"{elapsed_days:.1f}" if elapsed_days is not None else "N/A"

    print(f"k15-ready rows            : {count}")
    if marker:
        print(f"Last trained at rows      : {last_trained_rows}  ({last_ts_str})")
    else:
        print(f"Last trained at rows      : {last_trained_rows}  (no marker — never trained)")
    print(f"Days since last train     : {elapsed_str}")
    print()

    # Emergency trigger display
    if emergency_trigger:
        print(f"Emergency trigger         : FIRED  ({emergency_detail})")
    else:
        print(f"Emergency trigger         : NOT FIRED  ({emergency_detail})")

    # Row trigger display
    if row_trigger:
        print(f"Row-based trigger         : FIRED  ({count} >= {last_trained_rows} + {_ROW_TRIGGER_DELTA})")
    else:
        print(f"Row-based trigger         : not fired  ({count} < {last_trained_rows} + {_ROW_TRIGGER_DELTA})")

    # Time trigger display
    if time_trigger:
        if elapsed_days is None:
            print(f"Time-based trigger        : FIRED  (never trained)")
        else:
            print(f"Time-based trigger        : FIRED  ({elapsed_days:.1f} days >= {_TIME_TRIGGER_DAYS})")
    else:
        print(f"Time-based trigger        : not fired  ({elapsed_days:.1f} days < {_TIME_TRIGGER_DAYS})")

    print()

    # Determine which trigger fired
    if args.force:
        print("Trigger: --force")
    elif emergency_trigger:
        print("Trigger: EMERGENCY")
    elif row_trigger:
        print("Trigger: ROW-BASED")
    elif time_trigger:
        print("Trigger: TIME-BASED")
    else:
        print("No trigger fired. Exiting without retraining.")
        print(f"  Current state: {count} k15-ready rows, {elapsed_str} days since last train.")
        sys.exit(0)

    # 7. Check min rows guard
    if count < _MIN_ROWS:
        print(f"Minimum row requirement not met: {count} < {_MIN_ROWS}. Refusing to retrain.")
        sys.exit(1)

    # 8. Build subprocess command
    cmd = [
        sys.executable, "scripts/train_calibrator.py",
        "--db", args.db,
        "--out", args.out,
        "--min-rows", str(_MIN_ROWS),
        "--window", str(min(count, _WINDOW)),
    ]

    # 9. Dry-run: print command and exit
    if args.dry_run:
        print(f"--dry-run: would run: {' '.join(cmd)}")
        sys.exit(0)

    # 10. Run subprocess
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        save_marker(trained_at_rows=count, total_rows=count)
        print("Retraining succeeded. Marker updated.")
    else:
        print(f"Retraining FAILED (exit code {result.returncode}). Marker NOT updated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
