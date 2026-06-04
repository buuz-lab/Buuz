# Crontab entry (Monday 2am — model fresh for Monday volume pickup):
# 0 2 * * 1 cd "/Users/ezrakornberg/Kronos V2" && source .env && python3 scripts/auto_retrain_regime.py >> logs/auto_retrain_regime.log 2>&1
#
# Retraining triggers (in priority order):
#   1. Row-based: +200 new candle_features rows since last train (~2 days at 96/day)
#   2. Time-based: 14 days elapsed since last train
#
# Rolling window: last 2000 qualifying rows (~21 days). Time-ordered split: last
# _HOLDOUT_SIZE rows held out for evaluation against the deployed model.
# Candidate only replaces deployed model when holdout Brier strictly improves.
#
# Minimum rows: 672 (7 days × 96 candles/day). Unlike the calibrator, this script
# inlines training (not a subprocess) so it can compare candidate vs deployed Brier
# before committing the save.
"""
Auto-retrain script for the Kronos V2 regime model.

Evaluates retraining triggers and, when any fires, trains a candidate
RegimeModel in-process, evaluates holdout Brier vs the deployed model,
and saves only when the candidate is strictly better.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from btc_kalshi_system.models.regime_model import RegimeModel, _FEATURE_ORDER

# ── Constants ─────────────────────────────────────────────────────────────────

_MARKER_PATH = "models/regime_last_trained.json"

_ROW_TRIGGER_DELTA = 200   # retrain when +200 new candle rows since last train (~2 days)
_TIME_TRIGGER_DAYS = 14    # retrain if 14 days elapsed since last train
_MIN_ROWS = 672            # refuse to retrain below this (7 days × 96 candles/day)
_WINDOW = 2000             # rolling window: last 2000 candles (~21 days)
_HOLDOUT_SIZE = 100        # held-out rows for candidate vs deployed comparison

_FEATURE_COLS = list(_FEATURE_ORDER)

_CANDLE_QUERY = """
SELECT {cols}, btc_direction
FROM candle_features
WHERE features_stale = 0
  AND btc_direction IS NOT NULL
  AND funding_rate IS NOT NULL
  AND cvd_velocity IS NOT NULL
  AND brti_momentum_5min IS NOT NULL
  AND funding_window_proximity IS NOT NULL
  AND large_print_direction IS NOT NULL
  AND atm_iv IS NOT NULL
ORDER BY candle_ts ASC
"""


# ── Helper functions ──────────────────────────────────────────────────────────

def get_qualifying_count(db_path: str) -> int:
    """Return number of candle_features rows that pass training filters."""
    if not Path(db_path).exists():
        sys.exit(f"Database not found: {db_path}")
    cols = ", ".join(_FEATURE_COLS)
    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            f"SELECT COUNT(*) FROM ({_CANDLE_QUERY.format(cols=cols)})"
        ).fetchone()[0]
    finally:
        conn.close()
    return int(count)


def load_dataset(db_path: str, max_rows: int | None = None) -> list[tuple]:
    """Load qualifying candle_features rows, optionally limited to most recent max_rows."""
    cols = ", ".join(_FEATURE_COLS)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(_CANDLE_QUERY.format(cols=cols)).fetchall()
    finally:
        conn.close()
    if max_rows is not None and len(rows) > max_rows:
        rows = rows[-max_rows:]
    return rows


def build_xy(rows: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) where y = btc_direction (1=up, 0=down)."""
    arr = np.array(rows, dtype=object)
    n_features = len(_FEATURE_COLS)
    X = arr[:, :n_features].astype(np.float64)
    y = arr[:, n_features].astype(int)
    return X, y


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(np.mean((proba - y_true) ** 2))


def load_marker() -> dict | None:
    """Read _MARKER_PATH; return None if missing or corrupt."""
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


def save_marker(trained_at_rows: int, total_rows: int, holdout_brier: float) -> None:
    """Write _MARKER_PATH with current training state."""
    p = Path(_MARKER_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "trained_at_rows": trained_at_rows,
        "trained_at_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_rows_at_train": total_rows,
        "holdout_brier": holdout_brier,
    }
    with p.open("w") as f:
        json.dump(data, f, indent=4)


def evaluate_deployed_model(
    model_path: str, X_holdout: np.ndarray, y_holdout: np.ndarray
) -> float | None:
    """Return holdout Brier of the deployed model, or None if no model exists."""
    if not Path(model_path).exists():
        return None
    try:
        m = RegimeModel.load(model_path)
        proba = m._clf.predict_proba(X_holdout)[:, 1]
        return brier_score(y_holdout.astype(float), proba)
    except Exception:
        return None


def should_retrain(count: int, marker: dict | None, force: bool = False) -> str | None:
    """
    Return trigger name ('FORCE', 'ROW-BASED', 'TIME-BASED') or None if no trigger fires.
    """
    if force:
        return "FORCE"

    last_trained_rows = marker["trained_at_rows"] if marker else 0
    if count >= last_trained_rows + _ROW_TRIGGER_DELTA:
        return "ROW-BASED"

    if marker:
        last_ts = datetime.fromisoformat(marker["trained_at_timestamp"])
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        elapsed_days = (datetime.now(timezone.utc) - last_ts).total_seconds() / 86400
        if elapsed_days >= _TIME_TRIGGER_DAYS:
            return "TIME-BASED"
    else:
        return "TIME-BASED"  # Never trained

    return None


def should_deploy(candidate_brier: float, deployed_brier: float | None) -> bool:
    """Return True if candidate model should replace the deployed model."""
    if deployed_brier is None:
        return True
    return candidate_brier < deployed_brier


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--db", default="trades.db")
    p.add_argument("--out", default=config.REGIME_MODEL_PATH,
                   help=f"Output path (default: {config.REGIME_MODEL_PATH})")
    p.add_argument("--force", action="store_true",
                   help="Bypass trigger checks and retrain unconditionally.")
    p.add_argument("--dry-run", action="store_true",
                   help="Evaluate triggers and print what would happen without saving.")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # 1. Get current qualifying row count
    count = get_qualifying_count(args.db)

    # 2. Load marker
    marker = load_marker()

    # 3. Evaluate triggers
    trigger = should_retrain(count=count, marker=marker, force=args.force)

    # 4. Print status
    last_trained_rows = marker["trained_at_rows"] if marker else 0
    last_ts_str = marker["trained_at_timestamp"] if marker else "never"
    print(f"Qualifying candle rows    : {count}")
    if marker:
        print(f"Last trained at rows      : {last_trained_rows}  ({last_ts_str})")
        last_ts = datetime.fromisoformat(last_ts_str)
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        elapsed_days = (datetime.now(timezone.utc) - last_ts).total_seconds() / 86400
        print(f"Days since last train     : {elapsed_days:.1f}")
    else:
        print(f"Last trained at rows      : never")
        print(f"Days since last train     : N/A")

    row_trigger = count >= last_trained_rows + _ROW_TRIGGER_DELTA
    if row_trigger:
        print(f"Row-based trigger         : FIRED  ({count} >= {last_trained_rows} + {_ROW_TRIGGER_DELTA})")
    else:
        print(f"Row-based trigger         : not fired  ({count} < {last_trained_rows} + {_ROW_TRIGGER_DELTA})")
    print()

    if trigger is None:
        print("No trigger fired. Exiting without retraining.")
        sys.exit(0)

    print(f"Trigger: {trigger}")

    # 5. Min rows guard
    if count < _MIN_ROWS:
        print(f"Minimum row requirement not met: {count} < {_MIN_ROWS}. Refusing to retrain.")
        sys.exit(1)

    # 6. Load dataset and split
    rows = load_dataset(args.db, max_rows=_WINDOW)
    n = len(rows)
    X, y = build_xy(rows)

    if n <= _HOLDOUT_SIZE + 50:
        print(f"Not enough rows for train+holdout: {n} rows, holdout={_HOLDOUT_SIZE}.")
        sys.exit(1)

    X_train, X_holdout = X[:-_HOLDOUT_SIZE], X[-_HOLDOUT_SIZE:]
    y_train, y_holdout = y[:-_HOLDOUT_SIZE], y[-_HOLDOUT_SIZE:]

    # 7. Train candidate model
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    if pos == 0 or neg == 0:
        print(f"Degenerate label distribution: pos={pos} neg={neg}. Refusing to retrain.")
        sys.exit(1)

    pos_frac = pos / (pos + neg)
    extra_kwargs: dict = {}
    if pos_frac < 0.35 or pos_frac > 0.65:
        extra_kwargs["scale_pos_weight"] = neg / pos
        print(f"Applying scale_pos_weight={extra_kwargs['scale_pos_weight']:.3f}")

    candidate = RegimeModel()
    candidate.train(X_train, y_train, **extra_kwargs)

    # 8. Evaluate candidate on holdout
    proba_candidate = candidate._clf.predict_proba(X_holdout)[:, 1]
    candidate_brier = brier_score(y_holdout.astype(float), proba_candidate)
    candidate_acc = float(((proba_candidate >= 0.5).astype(int) == y_holdout).mean())
    print(f"Candidate holdout  Brier={candidate_brier:.4f}  Acc={candidate_acc:.4f}")

    # 9. Evaluate deployed model on same holdout
    deployed_brier = evaluate_deployed_model(args.out, X_holdout, y_holdout)
    if deployed_brier is not None:
        print(f"Deployed  holdout  Brier={deployed_brier:.4f}")
    else:
        print(f"Deployed  holdout  Brier=N/A (no model deployed)")

    deploy = should_deploy(candidate_brier, deployed_brier)
    if deploy:
        print(f"Holdout guard: PASS  (candidate {candidate_brier:.4f} < deployed "
              f"{deployed_brier:.4f if deployed_brier is not None else 'N/A'})")
    else:
        print(f"Holdout guard: FAIL  (candidate {candidate_brier:.4f} >= deployed {deployed_brier:.4f})")

    if args.dry_run:
        print("\n--dry-run: model NOT saved.")
        return

    if not deploy:
        print("Deployed model is better or equal. Not saving candidate.")
        sys.exit(0)

    # 10. Save candidate
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    candidate.save(str(out_path))
    save_marker(trained_at_rows=count, total_rows=n, holdout_brier=candidate_brier)
    print(f"\nSaved regime model → {out_path}")
    print(f"Marker updated: trained_at_rows={count}  holdout_brier={candidate_brier:.4f}")
    print("Restart KronosV2 to pick up the new model.")


if __name__ == "__main__":
    main()
