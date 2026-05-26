# Suggested crontab entry (runs every 6 hours — well within both trigger windows):
# 0 */6 * * * cd /path/to/KronosV2 && python3 scripts/auto_retrain.py >> logs/auto_retrain.log 2>&1
#
# Retraining triggers (in priority order):
#   1. Emergency: rolling accuracy on last 100 resolved trades < 55%
#   2. Row-based: +500 new training-ready rows since last train (~every 5 days)
#   3. Time-based: 14 days elapsed since last train (catches volume dry spells)
#
# Rolling window: uses all data until 1500 rows, then switches to last 1200 rows.
# Rationale: crypto regimes shift on a weeks timescale; old data hurts more than helps
# once the model has enough recent signal to work with.
"""
Auto-retrain script for the Kronos V2 regime model.

Evaluates retraining triggers and, when any fires, invokes train_regime.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from btc_kalshi_system.models.regime_model import RegimeModel, _FEATURE_ORDER

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

_MARKER_PATH = "models/last_trained.json"

_FRESH_FILTER = (
    "features_stale = 0"
    " AND funding_rate IS NOT NULL"
    " AND funding_rate_trend IS NOT NULL"
    " AND oi_delta_pct IS NOT NULL"
    " AND cvd_normalized IS NOT NULL"
    " AND basis_spread_pct IS NOT NULL"
    " AND brti_volatility_1h IS NOT NULL"
)

# Requires deribit_stale=0 so the row trigger fires only when enough 27-feature
# rows exist. Without this, the trigger would count ~388 rows (all 21-feature
# rows) and fire on ~June 2 when train_regime.py would immediately fail because
# it enforces deribit_stale=0 internally via _EXTRA_FILTERS_27.
_TRAINING_READY_FILTER = (
    "features_stale = 0"
    " AND funding_rate IS NOT NULL"
    " AND outcome IS NOT NULL"
    " AND deribit_stale = 0"
    " AND atm_iv IS NOT NULL"
)

_ROW_TRIGGER_DELTA = 200          # retrain when +200 new training-ready rows
_TIME_TRIGGER_DAYS = 14           # retrain if 14 days elapsed
_ROLLING_WINDOW_THRESHOLD = 1500  # switch to rolling window above this many rows
_ROLLING_WINDOW_SIZE = 1200       # use last 1200 rows when above threshold
_MIN_ROWS = 500                   # don't retrain if below this
_EMERGENCY_ACCURACY_THRESHOLD = 0.55


# ── Helper functions ──────────────────────────────────────────────────────────

def get_training_ready_count(db_path: str) -> int:
    """Return COUNT(*) of training-ready rows in the database."""
    if not Path(db_path).exists():
        sys.exit(f"Database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            f"SELECT COUNT(*) FROM trades WHERE {_TRAINING_READY_FILTER}"
        ).fetchone()[0]
    finally:
        conn.close()
    return int(count)


def compute_rolling_health(db_path: str, model_path: str) -> tuple[float, float] | None:
    """
    Load the deployed model and evaluate it on the last 100 resolved
    training-ready trades.  Returns (accuracy, brier) or None if no model file
    exists.
    """
    if not Path(model_path).exists():
        return None

    model = RegimeModel.load(model_path)

    if not Path(db_path).exists():
        return None

    feat_cols_sql = ", ".join(_FEATURE_ORDER)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            f"""SELECT {feat_cols_sql}, direction, outcome
                FROM trades
                WHERE {_TRAINING_READY_FILTER}
                ORDER BY timestamp DESC
                LIMIT 100"""
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    accuracies = []
    briers = []

    for row in rows:
        feat_vals = row[: len(_FEATURE_ORDER)]
        direction = int(row[len(_FEATURE_ORDER)])
        outcome = int(row[len(_FEATURE_ORDER) + 1])

        feat_dict = dict(zip(_FEATURE_ORDER, feat_vals))
        result = model.get_regime(feat_dict)

        prob_up = result["prob_up"]
        model_direction = result["direction"]

        up_label = 1 if (direction == outcome) else 0

        accuracies.append(int(model_direction == up_label))
        briers.append((prob_up - up_label) ** 2)

    accuracy = float(np.mean(accuracies))
    brier = float(np.mean(briers))
    return accuracy, brier


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
    p.add_argument("--out", default=config.REGIME_MODEL_PATH,
                   help=f"Output path for trained model (default: {config.REGIME_MODEL_PATH})")
    p.add_argument("--force", action="store_true",
                   help="Bypass all trigger checks and retrain unconditionally.")
    p.add_argument("--dry-run", action="store_true",
                   help="Evaluate triggers and print what would happen without retraining.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 1. Get current training-ready row count
    count = get_training_ready_count(args.db)

    # 2. Load marker
    marker = load_marker()

    # 3. Evaluate EMERGENCY trigger
    health = compute_rolling_health(args.db, config.REGIME_MODEL_PATH)
    emergency_trigger = health is not None and health[0] < _EMERGENCY_ACCURACY_THRESHOLD

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

    print(f"Training-ready rows       : {count}")
    if marker:
        print(f"Last trained at rows      : {last_trained_rows}  ({last_ts_str})")
    else:
        print(f"Last trained at rows      : {last_trained_rows}  (no marker — never trained)")
    print(f"Days since last train     : {elapsed_str}")
    print()

    # Emergency trigger display
    if emergency_trigger:
        emerg_detail = f"accuracy {health[0]:.2f} < {_EMERGENCY_ACCURACY_THRESHOLD}"
        print(f"Emergency trigger         : FIRED  ({emerg_detail})")
    else:
        if health is None:
            emerg_detail = "health OK or no model deployed"
        else:
            emerg_detail = f"accuracy {health[0]:.2f} >= {_EMERGENCY_ACCURACY_THRESHOLD}"
        print(f"Emergency trigger         : NOT FIRED  ({emerg_detail})")

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
        print(f"  Current state: {count} training-ready rows, {elapsed_str} days since last train.")
        sys.exit(0)

    # 7. Check min rows guard
    if count < _MIN_ROWS:
        print(f"Minimum row requirement not met: {count} < {_MIN_ROWS}. Refusing to retrain.")
        sys.exit(1)

    # 8. Build subprocess command
    cmd = [
        sys.executable, "scripts/train_regime.py",
        "--db", args.db,
        "--out", args.out,
        "--min-rows", str(_MIN_ROWS),
    ]
    if count > _ROLLING_WINDOW_THRESHOLD:
        cmd += ["--max-rows", str(_ROLLING_WINDOW_SIZE)]
        print(f"Rolling window active: using last {_ROLLING_WINDOW_SIZE} rows "
              f"(total training-ready: {count} > {_ROLLING_WINDOW_THRESHOLD})")

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
