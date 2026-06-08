# Suggested crontab entry (runs every 2 hours):
# 0 */2 * * * cd "/Users/ezrakornberg/Kronos V2" && python3 scripts/auto_retrain_mid_candle.py >> logs/auto_retrain_mid_candle.log 2>&1
#
# Retraining triggers (in priority order):
#   1. Row-based: +50 new qualifying rows since last train
#   2. Time-based: 7 days elapsed since last train
#
# Minimum rows: 200 qualifying mid-candle snapshots.
# Holdout guard: candidate Brier must strictly beat deployed model on the same 30 rows.
# Deploy gate: tails (top/bottom 20% prob) must outperform middle 60% by ≥5%.
"""
Auto-retrain script for the Kronos V2 mid-candle model.

Evaluates retraining triggers and, when any fires, trains a candidate
MidCandleModel in-process, evaluates holdout Brier + tail profitability,
and saves only when the candidate passes both gates.
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
from btc_kalshi_system.models.mid_candle_model import MidCandleModel, _MID_CANDLE_FEATURES

# ── Constants ─────────────────────────────────────────────────────────────────

_MARKER_PATH = "models/mid_candle_last_trained.json"

_ROW_TRIGGER_DELTA = 50   # retrain when +50 new qualifying rows since last train
_TIME_TRIGGER_DAYS = 7    # retrain if 7 days elapsed since last train
_MIN_ROWS = 200           # refuse to retrain below this
_HOLDOUT_SIZE = 30        # held-out rows for candidate vs deployed comparison

_CANDLE_QUERY = """
SELECT {cols}, btc_direction
FROM candle_features
WHERE btc_direction IS NOT NULL
  AND cvd_since_open IS NOT NULL
  AND kalshi_mid_candle_mid IS NOT NULL
  AND tick_count_since_open > 200
ORDER BY candle_ts ASC
"""

_FEATURE_COLS = list(_MID_CANDLE_FEATURES)


# ── Helper functions ──────────────────────────────────────────────────────────

def get_qualifying_count(db_path: str) -> int:
    cols = ", ".join(_FEATURE_COLS)
    conn = sqlite3.connect(db_path)
    try:
        return int(conn.execute(
            f"SELECT COUNT(*) FROM ({_CANDLE_QUERY.format(cols=cols)})"
        ).fetchone()[0])
    finally:
        conn.close()


def load_dataset(db_path: str) -> list[tuple]:
    cols = ", ".join(_FEATURE_COLS)
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(_CANDLE_QUERY.format(cols=cols)).fetchall()
    finally:
        conn.close()
    return rows


def build_xy(rows: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.array(rows, dtype=object)
    X = arr[:, : len(_FEATURE_COLS)].astype(np.float64)
    y = arr[:, len(_FEATURE_COLS)].astype(int)
    return X, y


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(np.mean((proba - y_true) ** 2))


def tails_beat_middle(proba: np.ndarray, y: np.ndarray) -> bool:
    """Return True if top/bottom 20% prob predictions outperform middle 60% by ≥5%."""
    if len(proba) < 20:
        return False
    lo = np.percentile(proba, 20)
    hi = np.percentile(proba, 80)
    tail_mask = (proba <= lo) | (proba >= hi)
    mid_mask  = ~tail_mask
    if tail_mask.sum() < 5 or mid_mask.sum() < 5:
        return False
    tail_acc = float(((proba[tail_mask] >= 0.5).astype(int) == y[tail_mask]).mean())
    mid_acc  = float(((proba[mid_mask]  >= 0.5).astype(int) == y[mid_mask]).mean())
    return tail_acc > mid_acc + 0.05


def load_marker() -> dict | None:
    p = Path(_MARKER_PATH)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        _ = data["trained_at_rows"], data["trained_at_timestamp"]
        return data
    except (json.JSONDecodeError, KeyError):
        print(f"WARNING: marker {_MARKER_PATH} corrupt — treating as absent.")
        return None


def save_marker(trained_at_rows: int, test_brier: float) -> None:
    data = {
        "trained_at_rows": trained_at_rows,
        "trained_at_timestamp": datetime.now(timezone.utc).isoformat(),
        "test_brier": test_brier,
    }
    Path(_MARKER_PATH).parent.mkdir(parents=True, exist_ok=True)
    Path(_MARKER_PATH).write_text(json.dumps(data, indent=4))


def evaluate_deployed(model_path: str, X: np.ndarray, y: np.ndarray) -> float | None:
    if not Path(model_path).exists():
        return None
    try:
        m = MidCandleModel.load(model_path)
        X_dict_list = [{_FEATURE_COLS[i]: float(X[j, i]) for i in range(len(_FEATURE_COLS))}
                       for j in range(len(X))]
        proba = np.array([m.predict(d)["prob_up"] for d in X_dict_list])
        return brier_score(y.astype(float), proba)
    except Exception as exc:
        print(f"Could not evaluate deployed model: {exc}")
        return None


def should_retrain(count: int, marker: dict | None, force: bool = False) -> str | None:
    if force:
        return "FORCE"
    last_rows = marker["trained_at_rows"] if marker else 0
    if count >= last_rows + _ROW_TRIGGER_DELTA:
        return "ROW-BASED"
    if marker:
        last_ts = datetime.fromisoformat(marker["trained_at_timestamp"])
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds() / 86400
        if elapsed >= _TIME_TRIGGER_DAYS:
            return "TIME-BASED"
    else:
        return "TIME-BASED"
    return None


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db",      default="trades.db")
    p.add_argument("--out",     default=config.MID_CANDLE_MODEL_PATH)
    p.add_argument("--force",   action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    count  = get_qualifying_count(args.db)
    marker = load_marker()
    trigger = should_retrain(count, marker, force=args.force)

    last_rows  = marker["trained_at_rows"] if marker else 0
    last_ts    = marker["trained_at_timestamp"] if marker else "never"
    print(f"Qualifying mid-candle rows : {count}")
    print(f"Last trained at rows       : {last_rows}  ({last_ts})")
    if count >= last_rows + _ROW_TRIGGER_DELTA:
        print(f"Row-based trigger          : FIRED  ({count} >= {last_rows} + {_ROW_TRIGGER_DELTA})")
    else:
        print(f"Row-based trigger          : not fired  ({count} < {last_rows} + {_ROW_TRIGGER_DELTA})")
    print()

    if trigger is None:
        print("No trigger fired. Exiting without retraining.")
        sys.exit(0)

    print(f"Trigger: {trigger}")

    if count < _MIN_ROWS:
        print(f"Minimum row requirement not met: {count} < {_MIN_ROWS}. Refusing to retrain.")
        sys.exit(1)

    rows = load_dataset(args.db)
    n = len(rows)
    X, y = build_xy(rows)

    if n <= _HOLDOUT_SIZE + 20:
        print(f"Not enough rows for train+holdout: {n} rows.")
        sys.exit(1)

    X_train, X_holdout = X[:-_HOLDOUT_SIZE], X[-_HOLDOUT_SIZE:]
    y_train, y_holdout = y[:-_HOLDOUT_SIZE], y[-_HOLDOUT_SIZE:]

    pos_frac = int(y_train.sum()) / len(y_train)
    extra: dict = {}
    if pos_frac < 0.35 or pos_frac > 0.65:
        extra["scale_pos_weight"] = (len(y_train) - int(y_train.sum())) / int(y_train.sum())
        print(f"Applying scale_pos_weight={extra['scale_pos_weight']:.3f}")

    candidate = MidCandleModel()
    candidate.train(X_train, y_train, **extra)

    # Evaluate candidate on holdout
    X_holdout_dicts = [{_FEATURE_COLS[i]: float(X_holdout[j, i])
                        for i in range(len(_FEATURE_COLS))} for j in range(len(X_holdout))]
    proba_holdout = np.array([candidate.predict(d)["prob_up"] for d in X_holdout_dicts])
    candidate_brier = brier_score(y_holdout.astype(float), proba_holdout)
    candidate_acc   = float(((proba_holdout >= 0.5).astype(int) == y_holdout).mean())
    print(f"Candidate holdout  Brier={candidate_brier:.4f}  Acc={candidate_acc:.4f}")

    # Tail profitability gate
    tails_ok = tails_beat_middle(proba_holdout, y_holdout)
    print(f"Tail profitability gate    : {'PASS ✓' if tails_ok else 'FAIL ✗ — tails do not outperform middle'}")

    # Holdout guard vs deployed
    deployed_brier = evaluate_deployed(args.out, X_holdout, y_holdout)
    if deployed_brier is not None:
        print(f"Deployed  holdout  Brier={deployed_brier:.4f}")
        deploy_ok = candidate_brier < deployed_brier
        print(f"Holdout guard              : {'PASS ✓' if deploy_ok else 'FAIL ✗ — deployed model better'}")
    else:
        print("Deployed  holdout  Brier=N/A (no model deployed)")
        deploy_ok = True

    if args.dry_run:
        print("\n--dry-run: model NOT saved.")
        return

    if not tails_ok:
        print("Deploy gate failed (tails). Not saving candidate.")
        sys.exit(0)

    if not deploy_ok:
        print("Deployed model is better. Not saving candidate.")
        sys.exit(0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    candidate.save(args.out)
    save_marker(trained_at_rows=count, test_brier=candidate_brier)
    print(f"\nSaved mid-candle model → {args.out}")
    print("Restart KronosV2 to pick up the new model.")


if __name__ == "__main__":
    main()
