"""
Regime model health check for Kronos V2.

Reports training progress, feature staleness, zero-variance warnings, and
post-deployment accuracy/Brier score for the live regime model (if deployed).

Usage:
    python3 scripts/regime_health_check.py [--db trades.db]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from btc_kalshi_system.models.regime_model import RegimeModel

_FEATURE_COLS = [
    "funding_rate",
    "funding_rate_trend",
    "oi_delta_pct",
    "cvd_normalized",
    "basis_spread_pct",
    "brti_volatility_1h",
    "cvd_velocity",
    "cvd_acceleration",
    "brti_momentum_5min",
    "brti_momentum_15min",
    "candle_progress",
    "hour_sin",
    "hour_cos",
    "kalshi_implied_prob",
    "funding_window_proximity",
    "trend_slope_1h",
    "trend_r2_1h",
    "hourly_sr_proximity",
    "range_breakout_flag",
    "tape_speed_tpm",
]

# Legacy 6-feature filter: used to count post-instrumentation rows (all rows since
# the original feature logging was added, regardless of 20-feature expansion).
_LEGACY_FILTER = (
    "features_stale = 0"
    " AND funding_rate IS NOT NULL"
    " AND funding_rate_trend IS NOT NULL"
    " AND oi_delta_pct IS NOT NULL"
    " AND cvd_normalized IS NOT NULL"
    " AND basis_spread_pct IS NOT NULL"
    " AND brti_volatility_1h IS NOT NULL"
)

# 20-feature filter: rows eligible for regime model training.
_FRESH_FILTER = (
    _LEGACY_FILTER
    + " AND cvd_velocity IS NOT NULL"
    + " AND brti_momentum_5min IS NOT NULL"
    + " AND kalshi_implied_prob IS NOT NULL"
    + " AND funding_window_proximity IS NOT NULL"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--db",
        default="trades.db",
        help="Path to trades.db (default: trades.db)",
    )
    return p.parse_args()


def _open(db_path: str) -> sqlite3.Connection:
    """Return a read-only sqlite3 connection, or exit with a clear message."""
    if not Path(db_path).exists():
        sys.exit(f"Database not found: {db_path}")
    return sqlite3.connect(db_path)


# ── Progress bar helper ────────────────────────────────────────────────────────

def _progress_bar(current: int, target: int, width: int = 20) -> str:
    pct = min(current / target, 1.0) if target > 0 else 0.0
    filled = int(round(pct * width))
    bar = "#" * filled + "_" * (width - filled)
    return f"[{bar}] {int(pct * 100)}%"


# ── Section 1 & 3: Training progress + zero-variance ─────────────────────────

def section_training_progress(conn: sqlite3.Connection) -> None:
    now = time.time()
    seven_days_ago = now - 7 * 86400

    # Counts
    total_rows = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    post_instr = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE funding_rate IS NOT NULL"
    ).fetchone()[0]
    training_ready_6 = conn.execute(
        f"""SELECT COUNT(*) FROM trades
           WHERE {_LEGACY_FILTER}
             AND outcome IS NOT NULL"""
    ).fetchone()[0]
    training_ready_20 = conn.execute(
        f"""SELECT COUNT(*) FROM trades
           WHERE {_FRESH_FILTER}
             AND outcome IS NOT NULL"""
    ).fetchone()[0]

    print("=== TRAINING PROGRESS ===")
    print(f"Total rows in trades.db        : {total_rows}")
    print(f"Post-instrumentation rows      : {post_instr}  (funding_rate IS NOT NULL)")
    print(f"Training-ready (6-feature)     : {training_ready_6}  (features_stale=0, resolved)")
    print(f"Training-ready (20-feature)    : {training_ready_20}  (all new cols NOT NULL)")
    print(f"Progress to 500 (20-feature)   : {_progress_bar(training_ready_20, 500)}")

    # Resolved rate over last 7 days (or full window if less data)
    recent_resolved = conn.execute(
        """SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
           FROM trades
           WHERE outcome IS NOT NULL
             AND timestamp >= ?""",
        (seven_days_ago,),
    ).fetchone()

    recent_count = recent_resolved[0] or 0
    if recent_count > 0:
        window_days = 7.0
        resolved_rate = recent_count / window_days
        window_label = "last 7 days"
    else:
        # Fall back to all resolved rows
        all_resolved = conn.execute(
            """SELECT COUNT(*), MIN(timestamp), MAX(timestamp)
               FROM trades WHERE outcome IS NOT NULL"""
        ).fetchone()
        all_count = all_resolved[0] or 0
        min_ts = all_resolved[1]
        max_ts = all_resolved[2]
        if all_count > 1 and min_ts is not None and max_ts is not None:
            span_days = max(max_ts - min_ts, 1) / 86400.0
            resolved_rate = all_count / span_days
            window_label = f"all data ({span_days:.1f} days)"
            recent_count = all_count
        else:
            resolved_rate = 0.0
            window_label = "no resolved data"
            recent_count = all_count

    print()
    print(f"Resolved rate ({window_label:<25s}): {resolved_rate:.1f} trades/day")
    if training_ready_20 >= 500:
        print("Estimated days to 500          : READY")
    elif resolved_rate > 0:
        days_needed = (500 - training_ready_20) / resolved_rate
        print(f"Estimated days to 500          : {days_needed:.1f}")
    else:
        print("Estimated days to 500          : N/A (no resolved-rate data)")

    # Feature variance stats on fresh rows
    feat_cols_sql = ", ".join(_FEATURE_COLS)
    fresh_rows = conn.execute(
        f"""SELECT {feat_cols_sql} FROM trades
            WHERE {_FRESH_FILTER}"""
    ).fetchall()

    print()
    print("Feature variance stats (fresh rows):")
    if not fresh_rows:
        print("  (no fresh rows available)")
    else:
        arr = np.array(fresh_rows, dtype=float)
        header = f"  {'Feature':<26s} {'mean':>12s} {'std':>12s} {'min':>12s} {'max':>12s}"
        print(header)
        for i, feat in enumerate(_FEATURE_COLS):
            col = arr[:, i]
            print(
                f"  {feat:<26s} {np.mean(col):>12.6f} {np.std(col):>12.6f}"
                f" {np.min(col):>12.6f} {np.max(col):>12.6f}"
            )

    print()


# ── Section 2: Feature staleness ─────────────────────────────────────────────

def section_feature_staleness(conn: sqlite3.Connection) -> None:
    print("=== FEATURE STALENESS ===")

    post_instr = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE funding_rate IS NOT NULL"
    ).fetchone()[0]

    stale = conn.execute(
        f"""SELECT COUNT(*) FROM trades
           WHERE funding_rate IS NOT NULL AND features_stale = 1"""
    ).fetchone()[0]

    fresh = conn.execute(
        f"""SELECT COUNT(*) FROM trades
           WHERE funding_rate IS NOT NULL AND features_stale = 0"""
    ).fetchone()[0]

    stale_pct = (stale / post_instr * 100) if post_instr > 0 else 0.0
    fresh_pct = (fresh / post_instr * 100) if post_instr > 0 else 0.0

    print(f"Post-instrumentation trades    : {post_instr}")
    print(f"Stale features (features_stale=1): {stale} ({stale_pct:.1f}%)")
    print(f"Fresh features (features_stale=0): {fresh} ({fresh_pct:.1f}%)")
    if stale + fresh < post_instr:
        print(f"  (Note: {post_instr - stale - fresh} rows have NULL features_stale — excluded from counts)")
    print()


# ── Section 3: Zero-variance check ───────────────────────────────────────────

def section_zero_variance(conn: sqlite3.Connection) -> None:
    print("=== ZERO-VARIANCE CHECK ===")

    feat_cols_sql = ", ".join(_FEATURE_COLS)
    fresh_rows = conn.execute(
        f"""SELECT {feat_cols_sql} FROM trades
            WHERE {_FRESH_FILTER}"""
    ).fetchall()

    if not fresh_rows:
        print("  (no fresh rows — cannot check variance)")
        print()
        return

    arr = np.array(fresh_rows, dtype=float)
    zero_var_found = False
    for i, feat in enumerate(_FEATURE_COLS):
        std = float(np.std(arr[:, i]))
        if std < 1e-6:
            print(
                f"  WARNING: '{feat}' has near-zero std (std={std:.2e})"
                " — pipeline may be stuck"
            )
            zero_var_found = True

    if not zero_var_found:
        print("  All features have non-zero variance. OK.")
    print()


# ── Section 4: Post-deployment health ────────────────────────────────────────

def section_post_deployment_health(conn: sqlite3.Connection, model_path: str) -> None:
    print("=== POST-DEPLOYMENT HEALTH ===")

    try:
        model = RegimeModel.load(model_path)
    except FileNotFoundError:
        print(f"No model deployed at {model_path} — skipping.")
        print()
        return

    print(f"Model found at: {model_path}")
    print("Evaluating on last 100 resolved training-ready trades...")

    feat_cols_sql = ", ".join(_FEATURE_COLS)
    rows = conn.execute(
        f"""SELECT {feat_cols_sql}, direction, outcome
            FROM trades
            WHERE {_FRESH_FILTER}
              AND outcome IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 100"""
    ).fetchall()

    if not rows:
        print("  No resolved training-ready trades found — cannot evaluate.")
        print()
        return

    n = len(rows)
    accuracies = []
    briers = []

    for row in rows:
        feat_vals = row[: len(_FEATURE_COLS)]
        direction = int(row[len(_FEATURE_COLS)])
        outcome = int(row[len(_FEATURE_COLS) + 1])

        feat_dict = dict(zip(_FEATURE_COLS, feat_vals))
        result = model.get_regime(feat_dict)

        prob_up = result["prob_up"]
        model_direction = result["direction"]

        # up_label=1 means the market actually went up
        up_label = 1 if (direction == outcome) else 0

        accuracies.append(int(model_direction == up_label))
        briers.append((prob_up - up_label) ** 2)

    rolling_accuracy = float(np.mean(accuracies))
    rolling_brier = float(np.mean(briers))

    print(f"Rolling accuracy (last {n})   : {rolling_accuracy:.2f}")
    print(f"Rolling Brier score (last {n}): {rolling_brier:.2f}   (0.25 = coin flip)")
    print()

    degraded = rolling_accuracy < 0.55 or rolling_brier > 0.25
    if degraded:
        reasons = []
        if rolling_accuracy < 0.55:
            reasons.append(f"accuracy {rolling_accuracy:.2f} < 0.55")
        if rolling_brier > 0.25:
            reasons.append(f"Brier {rolling_brier:.2f} > 0.25")
        print(f"STATUS: DEGRADED  ({', '.join(reasons)})")
    else:
        print("STATUS: OK")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    conn = _open(args.db)
    try:
        # Guard against completely empty / schema-less DB
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "trades" not in tables:
            print("trades table does not exist in the database — nothing to report.")
            return

        section_training_progress(conn)
        section_feature_staleness(conn)
        section_zero_variance(conn)
        section_post_deployment_health(conn, config.REGIME_MODEL_PATH)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
